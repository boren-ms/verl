"""Balanced offline augmentation of whole utterances before task formatting."""

import math
import random
from collections import Counter, defaultdict


def plan_expansion(rows, *, seed=42, rounds=4, max_audio_length=180, excluded_groups=()):
    """Return (ordered indices, round, anchor) with distinct partners.

    Alternating pair/triple rounds each contain N parents. Circular permutations
    balance every position exactly. Disjoint signed offsets prevent any pair of
    originals from co-occurring twice, even in a different order or triple.
    This intentionally fails if the requested diversity or duration is infeasible.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2 or rounds % 2:
        raise ValueError("rounds must be a positive even integer >= 2")
    n = len(rows)
    if n < 4 * rounds + 1:
        raise ValueError("not enough originals for distinct partners across rounds")
    if not math.isfinite(max_audio_length) or max_audio_length <= 0:
        raise ValueError("max_audio_length must be finite and positive")
    durations = [float(row["audio_length"]) for row in rows]
    paths = [row["audio_path"] for row in rows]
    if any(not isinstance(path, str) or not path or ";" in path for path in paths):
        raise ValueError("expected individual original audio paths")
    if len(set(paths)) != n:
        raise ValueError("original audio paths must be unique")
    if any(not math.isfinite(d) or d <= 0 for d in durations):
        raise ValueError("original durations must be finite and positive")
    if sum(sorted(durations)[:3]) >= max_audio_length:
        raise ValueError("duration cap cannot accommodate a triple")
    rng = random.Random(seed)
    permutation = list(range(n))
    rng.shuffle(permutation)
    excluded = {tuple(sorted(group)) for group in excluded_groups}
    used_differences, plans, offsets_by_round = set(), [], []
    for round_index in range(rounds):
        count = 2 + round_index % 2
        for _ in range(10000):
            offsets = [0, *rng.sample(range(1, n), count - 1)]
            differences = {(a - b) % n for a in offsets for b in offsets if a != b}
            if len(differences) != count * (count - 1) or differences & used_differences:
                continue
            groups = []
            for i in range(n):
                members = tuple(permutation[(i + offset) % n] for offset in offsets)
                if round_index % 2:
                    members = members[::-1]
                if tuple(sorted(members)) in excluded or math.fsum(durations[j] for j in members) >= max_audio_length:
                    break
                groups.append((members, round_index, permutation[i]))
            else:
                plans.extend(groups)
                used_differences.update(differences)
                offsets_by_round.append(offsets)
                break
        else:
            raise ValueError("cannot find balanced distinct partners within duration/exclusion constraints")
    rng.shuffle(plans)
    return plans, offsets_by_round


def plan_random_expansion(rows, *, seed=43, rounds=4, merge_count=(1, 5), max_audio_length=180, excluded_groups=()):
    """Random variable-size parents with exact size quotas and balanced anchors.

    Each original anchors exactly ``rounds`` parents. Singletons are sampled
    without replacement; partners and anchor positions are random. Repeated
    unordered groups and repeated pairwise partnerships are rejected. Unlike
    the circular pair/triple design, component reuse/positions are variable and
    must be reported from actual output counts rather than assumed balanced.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if (
        len(merge_count) != 2
        or any(isinstance(k, bool) or not isinstance(k, int) for k in merge_count)
        or not 1 <= merge_count[0] <= merge_count[1]
    ):
        raise ValueError("merge_count must contain ordered positive integers")
    n = len(rows)
    low, high = merge_count
    total, width = rounds * n, high - low + 1
    if n < high or total % width:
        raise ValueError("pool too small or output count not divisible by merge-count range")
    paths = [row["audio_path"] for row in rows]
    if any(not isinstance(p, str) or not p or ";" in p for p in paths) or len(set(paths)) != n:
        raise ValueError("expected unique individual original audio paths")
    durations = [float(row["audio_length"]) for row in rows]
    if (
        not math.isfinite(max_audio_length)
        or max_audio_length <= 0
        or any(not math.isfinite(d) or d <= 0 for d in durations)
    ):
        raise ValueError("durations and max_audio_length must be finite and positive")
    if math.fsum(sorted(durations)[:high]) >= max_audio_length:
        raise ValueError("duration cap cannot accommodate the maximum merge count")
    rng = random.Random(seed)
    quota = total // width
    excluded = {tuple(sorted(group)) for group in excluded_groups}
    plans, anchored, used_pairs, generated = [], Counter(), defaultdict(set), set()
    if low == 1:
        eligible = [i for i in range(n) if (i,) not in excluded and durations[i] < max_audio_length]
        if quota > len(eligible):
            raise ValueError("not enough unique eligible originals for singleton quota")
        for index in rng.sample(eligible, quota):
            plans.append(((index,), 0, index))
            anchored[index] += 1
            generated.add((index,))
    anchors = [i for i in range(n) for _ in range(rounds - anchored[i])]
    counts = [count for count in range(max(2, low), high + 1) for _ in range(quota)]
    rng.shuffle(anchors)
    rng.shuffle(counts)
    for anchor, count in zip(anchors, counts, strict=True):
        for _ in range(10000):
            members = rng.sample(range(n), count - 1)
            if anchor in members:
                continue
            members.insert(rng.randrange(count), anchor)
            signature = tuple(sorted(members))
            if signature in excluded or signature in generated:
                continue
            if any(b in used_pairs[a] for pos, a in enumerate(members) for b in members[pos + 1 :]):
                continue
            if math.fsum(durations[i] for i in members) >= max_audio_length:
                continue
            plans.append((tuple(members), anchored[anchor], anchor))
            anchored[anchor] += 1
            generated.add(signature)
            for index in members:
                used_pairs[index].update(i for i in members if i != index)
            break
        else:
            raise ValueError("cannot satisfy random merge duration/diversity constraints")
    assert len(plans) == total and set(anchored.values()) == {rounds} and len(anchored) == n
    rng.shuffle(plans)
    return plans
