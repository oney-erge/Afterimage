"""CPU-only structural representation for a speculative candidate tree,
built ahead of H20 (real tree-attention verification) so the STRUCTURE a
verifier will eventually consume can be validated and reasoned about
before any GPU-side masking code exists.

This module deliberately implements NONE of the following, per
docs/SPECULATION_TREE_RESEARCH.md's own exactness-boundary rules and this
project's "no scaffolding for a future feature" principle
(afterimage/runtime/config.py): tree-attention masking, a real target
verifier, a draft-tree planner, or any CUDA path. SpecTree is a plain data
structure with structural invariants a planner or verifier can be built
against later -- what it is NOT is a claim that either of those exists
yet. See docs/SPECULATION_TREE_RESEARCH.md's implementation-order section
for what this unblocks and what it does not.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SpecNode:
    """One candidate token in a speculative tree.

    source_prob is the proposing source's own probability for this token
    given its parent path (e.g. a draft model's P(token | prefix));
    posterior_prob is filled in later by whatever combines multiple
    sources' beliefs (H25's ScenarioSpec/PosteriorTree line) and is None
    until that exists. probe marks a node inserted purely to gather
    information (H28's ProbeSpec) rather than to propose a real
    continuation -- kept as a first-class field now so a future probe
    mechanism does not require restructuring this type.
    """
    node_id: int
    parent_id: int | None
    token_id: int
    depth: int
    source: str
    source_prob: float | None
    posterior_prob: float | None = None
    probe: bool = False


class SpecTreeError(ValueError):
    """A SpecTree failed structural validation. Distinct from plain
    ValueError so a caller can catch tree-specific integrity failures
    without also swallowing unrelated argument errors."""


class SpecTree:
    """A validated, immutable-after-construction collection of SpecNodes
    forming a tree (a single root, no cycles, every non-root node's
    parent already present).

    Construction does NOT validate automatically -- call validate()
    explicitly. This is deliberate: a caller building a tree incrementally
    (e.g. a planner appending nodes one at a time) should be able to hold
    a temporarily-invalid intermediate state without every mutation
    re-running full validation, then validate once before handing the
    tree to anything that trusts its invariants.
    """

    def __init__(self, nodes: list[SpecNode]):
        self._nodes: dict[int, SpecNode] = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise SpecTreeError("duplicate node_id in constructor input")
        self._children: dict[int | None, list[int]] = {}
        for node in nodes:
            self._children.setdefault(node.parent_id, []).append(node.node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def node(self, node_id: int) -> SpecNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> list[SpecNode]:
        return list(self._nodes.values())

    def budget(self) -> int:
        """Total node count -- the quantity every node-budget constraint
        (this project's own measured N_free from H19, or a planner's
        configured cap) is actually counting against."""
        return len(self._nodes)

    def children(self, node_id: int | None) -> list[SpecNode]:
        """Direct children of node_id. Pass None for the root(s)."""
        return [self._nodes[cid] for cid in self._children.get(node_id, [])]

    def path_to(self, node_id: int) -> list[SpecNode]:
        """Root-to-node_id path, inclusive of both endpoints, in root-
        first order -- what a verifier needs to reconstruct the actual
        token sequence a given leaf represents."""
        if node_id not in self._nodes:
            raise KeyError("no such node_id: %r" % node_id)
        path = []
        current: int | None = node_id
        seen: set[int] = set()
        while current is not None:
            if current in seen:
                raise SpecTreeError(
                    "cycle detected while walking to root from node_id=%r" % node_id)
            seen.add(current)
            node = self._nodes[current]
            path.append(node)
            current = node.parent_id
        path.reverse()
        return path

    def validate(self) -> None:
        """Checks every structural invariant a downstream verifier is
        entitled to assume without re-checking itself:

        - every node_id is unique (guaranteed by construction already,
          re-checked here so validate() is a complete, self-contained
          contract rather than depending on constructor behavior)
        - every non-root node's parent_id refers to a node that IS
          present in this tree
        - no cycles (a node cannot be its own ancestor)
        - depth is consistent: depth == 0 for a root, depth ==
          parent.depth + 1 for every other node
        - there is exactly one root's worth of depth-0 nodes reachable
          from any given node's ancestry (implied by the no-cycle check
          plus every parent existing, but confirmed explicitly by walking
          every node to its root and requiring that walk to terminate)

        Raises SpecTreeError on the first violation found, with the
        offending node_id, rather than silently accepting a malformed
        tree that would fail confusingly later inside a verifier.
        """
        for node in self._nodes.values():
            if node.parent_id is not None and node.parent_id not in self._nodes:
                raise SpecTreeError(
                    "node_id=%r has parent_id=%r, which is not present in this tree" %
                    (node.node_id, node.parent_id))
            if node.parent_id is None:
                if node.depth != 0:
                    raise SpecTreeError(
                        "root node_id=%r must have depth=0, got depth=%r" %
                        (node.node_id, node.depth))
            else:
                parent = self._nodes[node.parent_id]
                if node.depth != parent.depth + 1:
                    raise SpecTreeError(
                        "node_id=%r has depth=%r but its parent (node_id=%r) has "
                        "depth=%r -- expected depth=%r" %
                        (node.node_id, node.depth, parent.node_id, parent.depth,
                         parent.depth + 1))
            # Cycle check: every node must reach a root (parent_id=None)
            # in at most len(self._nodes) hops, or it is part of a cycle.
            current: int | None = node.node_id
            hops = 0
            while current is not None:
                hops += 1
                if hops > len(self._nodes):
                    raise SpecTreeError(
                        "cycle detected reachable from node_id=%r" % node.node_id)
                current = self._nodes[current].parent_id

    def merge_prefixes(self, other: "SpecTree") -> "SpecTree":
        """Merges two trees that share a common token-sequence prefix
        (e.g. two candidate sources proposing continuations from the same
        already-accepted context) into one tree with that prefix
        represented once, not duplicated -- the real cost saving a tree-
        based scheme is supposed to realize over independently verifying
        each source's full path.

        A shared prefix is identified purely structurally, by walking
        both trees' roots and matching consecutive (token_id, source)
        pairs -- node_id values are NOT assumed to agree between the two
        input trees (they come from independent sources and have no
        reason to share an ID space). Node identity in the OUTPUT tree is
        freshly assigned, contiguous from 0, in the order nodes are
        added (shared prefix first, then self's remaining unique nodes,
        then other's remaining unique nodes) -- callers must not assume
        merged node_ids relate to either input's own IDs.

        Both inputs are assumed already valid (call validate() on each
        first); this method does not re-validate its inputs, only the
        merge logic's own output before returning it.
        """
        self_roots = self.children(None)
        other_roots = other.children(None)
        if len(self_roots) != 1 or len(other_roots) != 1:
            raise SpecTreeError(
                "merge_prefixes requires exactly one root per tree, got %d and %d" %
                (len(self_roots), len(other_roots)))

        merged_nodes: list[SpecNode] = []
        next_id = 0

        def _add(node: SpecNode, new_parent_id: int | None) -> int:
            nonlocal next_id
            merged_nodes.append(dataclasses.replace(
                node, node_id=next_id, parent_id=new_parent_id))
            assigned = next_id
            next_id += 1
            return assigned

        # Walk the shared prefix: as long as both trees have a single
        # child at the current position with matching (token_id, source),
        # merge them into one node. The moment they diverge (different
        # token, different source, or either side branches into more than
        # one child), the shared-prefix walk stops and each remaining
        # subtree is copied in independently under the last shared node.
        self_cursor: SpecNode | None = self_roots[0]
        other_cursor: SpecNode | None = other_roots[0]
        merged_parent_id: int | None = None
        while (self_cursor is not None and other_cursor is not None
               and self_cursor.token_id == other_cursor.token_id
               and self_cursor.source == other_cursor.source):
            merged_parent_id = _add(self_cursor, merged_parent_id)
            self_children = self.children(self_cursor.node_id)
            other_children = other.children(other_cursor.node_id)
            if len(self_children) == 1 and len(other_children) == 1:
                self_cursor, other_cursor = self_children[0], other_children[0]
            else:
                self_cursor, other_cursor = None, None

        def _copy_subtree(tree: "SpecTree", subtree_root: SpecNode | None,
                          new_parent_id: int | None) -> None:
            if subtree_root is None:
                return
            assigned = _add(subtree_root, new_parent_id)
            for child in tree.children(subtree_root.node_id):
                _copy_subtree(tree, child, assigned)

        # Whatever did not get folded into the shared-prefix walk above is
        # copied in as-is: self_cursor/other_cursor point at the first
        # diverging nodes (or None, if one tree's path was a pure prefix
        # of the other's and fully consumed).
        if self_cursor is not None:
            _copy_subtree(self, self_cursor, merged_parent_id)
        else:
            for child in (self.children(self_roots[0].node_id) if merged_parent_id is None
                         else []):
                _copy_subtree(self, child, merged_parent_id)
        if other_cursor is not None:
            _copy_subtree(other, other_cursor, merged_parent_id)
        else:
            for child in (other.children(other_roots[0].node_id) if merged_parent_id is None
                         else []):
                _copy_subtree(other, child, merged_parent_id)

        merged = SpecTree(merged_nodes)
        merged.validate()
        return merged
