"""SpecNode/SpecTree is a CPU-only structural representation with no
CUDA and no real verifier behind it (see afterimage/runtime/spec_tree.py's
module docstring for why that split is deliberate). What matters here is
purely structural correctness: no cycles, every parent present before its
child references it, depth bookkeeping stays consistent, node IDs stay
unique, budget is just node count, and prefix merging actually removes
duplication instead of silently keeping two copies of a shared path.
"""
from __future__ import annotations

import pytest

from afterimage.runtime.spec_tree import SpecNode, SpecTree, SpecTreeError


def _node(node_id, parent_id, token_id, depth, source="primary", prob=0.5):
    return SpecNode(node_id=node_id, parent_id=parent_id, token_id=token_id,
                    depth=depth, source=source, source_prob=prob)


# ------------------------------------------------------------------- construction

def test_constructor_rejects_duplicate_node_ids():
    with pytest.raises(SpecTreeError, match="duplicate node_id"):
        SpecTree([_node(0, None, 1, 0), _node(0, None, 2, 0)])


def test_len_and_budget_report_node_count():
    tree = SpecTree([_node(0, None, 1, 0), _node(1, 0, 2, 1), _node(2, 0, 3, 1)])
    assert len(tree) == 3
    assert tree.budget() == 3


def test_node_looks_up_by_id():
    tree = SpecTree([_node(0, None, 42, 0)])
    assert tree.node(0).token_id == 42


# ---------------------------------------------------------------------- children

def test_children_of_root_and_leaves():
    tree = SpecTree([
        _node(0, None, 1, 0),
        _node(1, 0, 2, 1),
        _node(2, 0, 3, 1),
        _node(3, 1, 4, 2),
    ])
    assert {n.node_id for n in tree.children(None)} == {0}
    assert {n.node_id for n in tree.children(0)} == {1, 2}
    assert tree.children(2) == []
    assert {n.node_id for n in tree.children(1)} == {3}


# ---------------------------------------------------------------------- path_to

def test_path_to_returns_root_first_order():
    tree = SpecTree([
        _node(0, None, 10, 0),
        _node(1, 0, 20, 1),
        _node(2, 1, 30, 2),
    ])
    path = tree.path_to(2)
    assert [n.node_id for n in path] == [0, 1, 2]
    assert [n.token_id for n in path] == [10, 20, 30]


def test_path_to_unknown_node_raises_key_error():
    tree = SpecTree([_node(0, None, 1, 0)])
    with pytest.raises(KeyError):
        tree.path_to(99)


def test_path_to_detects_a_cycle_rather_than_looping_forever():
    # node 0's parent is node 1, node 1's parent is node 0: a cycle with
    # no reachable root. Constructed directly (bypassing validate()) to
    # exercise path_to's own independent cycle guard.
    tree = SpecTree([_node(0, 1, 1, 0), _node(1, 0, 2, 0)])
    with pytest.raises(SpecTreeError, match="cycle detected"):
        tree.path_to(0)


# ----------------------------------------------------------------------- validate

def test_validate_accepts_a_well_formed_tree():
    tree = SpecTree([
        _node(0, None, 1, 0),
        _node(1, 0, 2, 1),
        _node(2, 0, 3, 1),
        _node(3, 1, 4, 2),
    ])
    tree.validate()  # must not raise


def test_validate_rejects_a_parent_that_does_not_exist():
    tree = SpecTree([_node(0, None, 1, 0), _node(1, 99, 2, 1)])
    with pytest.raises(SpecTreeError, match="not present in this tree"):
        tree.validate()


def test_validate_rejects_a_root_with_nonzero_depth():
    tree = SpecTree([_node(0, None, 1, 5)])
    with pytest.raises(SpecTreeError, match="must have depth=0"):
        tree.validate()


def test_validate_rejects_inconsistent_child_depth():
    tree = SpecTree([_node(0, None, 1, 0), _node(1, 0, 2, 5)])  # should be depth=1
    with pytest.raises(SpecTreeError, match="expected depth=1"):
        tree.validate()


def test_validate_rejects_a_two_node_cycle():
    # Depths are chosen to be LOCALLY consistent with each node's own
    # claimed parent (node 0 says its parent is node 1 with depth 1, so
    # node 0's depth=2 passes that check in isolation) specifically so
    # this test exercises the cycle-walk guard itself, not the (also
    # real, but different) depth-consistency check that would otherwise
    # fire first on a naively mismatched cycle.
    tree = SpecTree([_node(0, 1, 1, 2), _node(1, 0, 2, 1)])
    with pytest.raises(SpecTreeError, match="cycle detected"):
        tree.validate()


def test_validate_rejects_a_self_referencing_node():
    tree = SpecTree([_node(0, 0, 1, 1)])
    with pytest.raises(SpecTreeError):
        tree.validate()


# ------------------------------------------------------------------- merge_prefixes

def _linear_tree(tokens, source="primary"):
    """A simple single-path tree: root -> tokens[0] -> tokens[1] -> ..."""
    nodes = []
    parent = None
    for depth, token in enumerate(tokens):
        node_id = depth
        nodes.append(_node(node_id, parent, token, depth, source=source))
        parent = node_id
    return SpecTree(nodes)


def test_merge_prefixes_deduplicates_a_fully_shared_path():
    a = _linear_tree([1, 2, 3])
    b = _linear_tree([1, 2, 3])
    merged = a.merge_prefixes(b)
    merged.validate()
    # Identical single paths, same source at every step -> collapses to
    # exactly 3 nodes, not 6.
    assert len(merged) == 3
    leaf = [n for n in merged.nodes if merged.children(n.node_id) == []]
    assert len(leaf) == 1


def test_merge_prefixes_keeps_disjoint_paths_separate():
    a = _linear_tree([1, 2, 3])
    b = _linear_tree([9, 8, 7])
    merged = a.merge_prefixes(b)
    merged.validate()
    # No shared prefix at all (roots differ) -> two full paths, one
    # merged root-parent slot each: 3 + 3 = 6 nodes total, two roots'
    # worth of content under a single synthetic entry point is not
    # created (merge_prefixes requires one root per input, output can
    # have divergent top-level nodes once the shared walk stops
    # immediately).
    assert len(merged) == 6


def test_merge_prefixes_shares_a_partial_prefix_and_branches_after():
    a = _linear_tree([1, 2, 3])       # root(1) -> 2 -> 3
    b = _linear_tree([1, 2, 99])      # root(1) -> 2 -> 99 (diverges at depth 2)
    merged = a.merge_prefixes(b)
    merged.validate()
    # Shared: token 1 (depth 0), token 2 (depth 1) -> 2 nodes merged once.
    # Then diverges into two depth-2 children: 3 and 99.
    assert len(merged) == 4
    root = merged.children(None)[0]
    assert root.token_id == 1
    depth1 = merged.children(root.node_id)
    assert len(depth1) == 1
    assert depth1[0].token_id == 2
    depth2_tokens = {n.token_id for n in merged.children(depth1[0].node_id)}
    assert depth2_tokens == {3, 99}


def test_merge_prefixes_treats_different_sources_as_not_shared():
    """Two nodes with the same token_id but a different proposing source
    are NOT the same candidate -- merging them would silently hide which
    source actually gets credit if the merged node is later verified."""
    a = _linear_tree([1, 2], source="primary")
    b = _linear_tree([1, 2], source="scout")
    merged = a.merge_prefixes(b)
    merged.validate()
    assert len(merged) == 4  # nothing shared; both full paths kept


def test_merge_prefixes_output_node_ids_are_contiguous_from_zero():
    a = _linear_tree([1, 2, 3])
    b = _linear_tree([1, 9])
    merged = a.merge_prefixes(b)
    ids = sorted(n.node_id for n in merged.nodes)
    assert ids == list(range(len(merged)))


# The tests above all use LINEAR trees, which is precisely the shape that
# cannot exercise a branching divergence. An earlier implementation walked
# the shared prefix in lockstep and abandoned BOTH subtrees the moment
# either side had anything other than exactly one child -- silently
# dropping real candidate nodes while still returning a tree that passed
# validate() (a 5-node merge came back with 1 node). The cases below are
# the shapes that catch that.

def test_merge_prefixes_keeps_every_node_when_divergence_is_a_branch():
    """A branches into two children where B has one. Nothing may be lost:
    the shared root plus all three distinct continuations survive."""
    a = SpecTree([_node(0, None, 1, 0), _node(1, 0, 2, 1), _node(2, 0, 3, 1)])
    b = SpecTree([_node(0, None, 1, 0), _node(1, 0, 4, 1)])
    merged = a.merge_prefixes(b)
    merged.validate()
    assert sorted(n.token_id for n in merged.nodes) == [1, 2, 3, 4]
    root = merged.roots()[0]
    assert {n.token_id for n in merged.children(root.node_id)} == {2, 3, 4}


def test_merge_prefixes_keeps_the_tail_when_one_path_is_a_pure_prefix():
    """A's whole path is a prefix of B's. B's extra depth must survive."""
    a = _linear_tree([1, 2])
    b = _linear_tree([1, 2, 3])
    merged = a.merge_prefixes(b)
    merged.validate()
    assert sorted(n.token_id for n in merged.nodes) == [1, 2, 3]
    assert len(merged) == 3


def test_merge_prefixes_never_loses_or_duplicates_nodes_across_random_shapes():
    """Property check over branching shapes: a merge can only ever
    collapse shared nodes, so the result must hold at least as many nodes
    as the larger input and at most the two inputs combined."""
    import random
    rng = random.Random(0)

    def _random_tree(size):
        nodes = [_node(0, None, rng.randint(1, 4), 0,
                       source=rng.choice(("primary", "scout")))]
        for i in range(1, size):
            parent = rng.randrange(i)
            nodes.append(_node(i, parent, rng.randint(1, 4), nodes[parent].depth + 1,
                               source=rng.choice(("primary", "scout"))))
        return SpecTree(nodes)

    for _ in range(200):
        a = _random_tree(rng.randint(1, 7))
        b = _random_tree(rng.randint(1, 7))
        a.validate()
        b.validate()
        merged = a.merge_prefixes(b)
        merged.validate()
        assert max(len(a), len(b)) <= len(merged) <= len(a) + len(b)


def test_merge_prefixes_is_closed_over_multi_root_results():
    """Two sources whose first tokens differ produce a legal multi-root
    forest, and that forest must itself be mergeable -- otherwise three
    sources could never be folded together pairwise."""
    a = _linear_tree([1, 2])
    b = _linear_tree([9, 8])
    forest = a.merge_prefixes(b)
    assert len(forest.roots()) == 2
    forest.validate()
    again = forest.merge_prefixes(a)  # a is already fully contained
    again.validate()
    assert len(again) == len(forest)  # nothing new to add, nothing lost


def test_merge_prefixes_recomputes_depth_at_the_merged_position():
    """A subtree grafted in keeps its structure but takes its depth from
    where it lands, so validate()'s depth rule still holds."""
    shallow = _linear_tree([5])
    deep = _linear_tree([7, 8, 9])
    merged = shallow.merge_prefixes(deep)
    merged.validate()  # would raise if depths were inherited blindly
    by_token = {n.token_id: n.depth for n in merged.nodes}
    assert by_token == {5: 0, 7: 0, 8: 1, 9: 2}
