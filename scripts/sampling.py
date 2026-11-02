"""Reproducible operation-balanced subsets with source-design diversity."""
from __future__ import annotations

from collections import Counter, defaultdict
import random


def sample_tasks(tasks, count, seed=42):
    """Select distinct tasks without changing task contents.

    Operation counts are balanced up to availability. Bipartite matching then
    maximizes distinct source designs for those quotas; if repeated sources
    are unavoidable, remaining tasks favor the least represented source.
    Input order does not affect a selection with the same seed.
    """
    tasks=list(tasks)
    if type(count) is not int or not 1<=count<=len(tasks):
        raise ValueError('--sample must be between 1 and the number of dataset tasks')
    if type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('Sampling seed must be a nonnegative signed64-bit integer')
    seen=set();by_operation=defaultdict(list);by_operation_source=defaultdict(lambda:defaultdict(list))
    source_for={}
    for task in sorted(tasks,key=lambda row:str(row.get('id',''))):
        cid=task.get('id');operation=task.get('operation')
        if not isinstance(cid,str) or not cid or cid in seen:
            raise ValueError('Sampling requires distinct nonempty task IDs')
        if not isinstance(operation,str) or not operation:
            raise ValueError('Sampling requires a nonempty operation for every task')
        source=task.get('source_id',task.get('group_id',cid))
        if not isinstance(source,(str,int)) or isinstance(source,bool) or str(source)=='':
            raise ValueError('Invalid sampling source/group ID')
        source=str(source);seen.add(cid);source_for[cid]=source
        by_operation[operation].append(task)
        by_operation_source[operation][source].append(task)
    rng=random.Random(seed)
    operations=sorted(by_operation);rng.shuffle(operations)
    quotas=Counter()
    for _ in range(count):
        available=[op for op in operations if quotas[op]<len(by_operation[op])]
        minimum=min(quotas[op] for op in available)
        # The shuffled operation order is the deterministic tie breaker.
        op=next(op for op in available if quotas[op]==minimum)
        quotas[op]+=1
    slots=[op for op in operations for _ in range(quotas[op])]
    adjacency={op:sorted(by_operation_source[op]) for op in operations}
    for values in adjacency.values():rng.shuffle(values)
    owners={};assigned={}

    def augment(slot,visited):
        for source in adjacency[slots[slot]]:
            if source in visited:continue
            visited.add(source)
            previous=owners.get(source)
            if previous is None or augment(previous,visited):
                owners[source]=slot;assigned[slot]=source
                return True
        return False

    order=list(range(len(slots)));rng.shuffle(order)
    order.sort(key=lambda i:len(adjacency[slots[i]]))
    for slot in order:augment(slot,set())
    chosen={};selected_ids=set();source_counts=Counter()
    for slot,source in sorted(assigned.items()):
        item=rng.choice(by_operation_source[slots[slot]][source])
        chosen[slot]=item;selected_ids.add(item['id']);source_counts[source]+=1
    for slot,op in enumerate(slots):
        if slot in chosen:continue
        candidates=[t for t in by_operation[op] if t['id'] not in selected_ids]
        minimum=min(source_counts[source_for[t['id']]] for t in candidates)
        candidates=[t for t in candidates if source_counts[source_for[t['id']]]==minimum]
        item=rng.choice(candidates);chosen[slot]=item
        selected_ids.add(item['id']);source_counts[source_for[item['id']]]+=1
    result=[chosen[i] for i in range(len(slots))]
    rng.shuffle(result)
    return result
