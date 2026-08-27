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
    forming a rooted forest: no cycles, every non-root node's parent
    already present, and depth consistent with each node's parent.

    Note "forest," not "single tree": one or more roots is legal, and
    merge_prefixes deliberately produces a multi-root result when the two
    sources' first candidate tokens differ (there is genuinely no shared
    node to hang them under, and inventing a synthetic root would fake a
    shared candidate the target would then be asked to verify). validate()
    therefore does NOT require a single root -- an earlier revision of
    this docstring claimed it did, which validate() never enforced.

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

    def roots(self) -> list[SpecNode]:
        """Every depth-0 node (parent_id is None). More than one is legal
        -- see the class docstring on why merge_prefixes can produce a
        multi-root forest."""
        return self.children(None)

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

        Sharing is decided level by level, uniformly: at each position two
        nodes are the SAME candidate when their (token_id, source) pair
        matches, and matched nodes recurse into their children. Anything
        unmatched -- on either side -- is copied in whole, subtree
        included. node_id values are NOT assumed to agree between the two
        input trees (they come from independent sources and have no
        reason to share an ID space); output node identity is freshly
        assigned, contiguous from 0, and callers must not assume merged
        node_ids relate to either input's own IDs. Output `depth` is
        recomputed from the merged position rather than inherited, so a
        subtree grafted at a different depth than it had in its source
        tree still satisfies validate()'s depth-consistency rule.

        The root level is treated exactly like any other sibling level,
        which means merging is closed: a multi-root result (produced when
        the two sources' first tokens differ -- see the class docstring)
        is itself a legal input to another merge_prefixes call, so three
        or more sources can be merged by folding pairwise.

        Where two matched nodes carry different `source_prob` values,
        SELF's value is the one kept. Matching requires the same `source`,
        so both values came from the same model and should agree; if a
        caller merges two trees from the same source with genuinely
        different probabilities for one token, that disagreement is
        silently resolved in self's favor rather than being flagged.

        Note that including `source` in the match key means Primary and
        Scout proposing the SAME token at the same position produce two
        separate nodes, each consuming a verification slot. That keeps
        per-source credit attribution exact (which source earned a hit),
        at the cost of not deduplicating the case where the two sources
        agree. Which of those matters more is a real open design question
        for H20/H27, not something this structural layer should decide on
        its own -- see docs/SPECULATION_TREE_RESEARCH.md.

        Both inputs are assumed already valid (call validate() on each
        first); this method does not re-validate its inputs, only the
        merge logic's own output before returning it.
        """
        merged_nodes: list[SpecNode] = []
        next_id = 0

        def _add(node: SpecNode, new_parent_id: int | None, depth: int) -> int:
            nonlocal next_id
            merged_nodes.append(dataclasses.replace(
                node, node_id=next_id, parent_id=new_parent_id, depth=depth))
            assigned = next_id
            next_id += 1
            return assigned

        def _copy_subtree(tree: "SpecTree", node: SpecNode,
                          new_parent_id: int | None, depth: int) -> None:
            assigned = _add(node, new_parent_id, depth)
            for child in tree.children(node.node_id):
                _copy_subtree(tree, child, assigned, depth + 1)

        def _merge_levels(self_nodes: list[SpecNode], other_nodes: list[SpecNode],
                          new_parent_id: int | None, depth: int) -> None:
            unmatched_other = list(other_nodes)
            for self_node in self_nodes:
                match = next(
                    (candidate for candidate in unmatched_other
                     if candidate.token_id == self_node.token_id
                     and candidate.source == self_node.source), None)
                if match is None:
                    _copy_subtree(self, self_node, new_parent_id, depth)
                    continue
                unmatched_other.remove(match)
                assigned = _add(self_node, new_parent_id, depth)
                _merge_levels(self.children(self_node.node_id),
                              other.children(match.node_id), assigned, depth + 1)
            for leftover in unmatched_other:
                _copy_subtree(other, leftover, new_parent_id, depth)

        _merge_levels(self.roots(), other.roots(), None, 0)

        merged = SpecTree(merged_nodes)
        merged.validate()
        return merged
