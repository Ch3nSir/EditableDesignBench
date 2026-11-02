"""No model, API or production data is needed for subset selection checks."""
from collections import Counter
from copy import deepcopy
import unittest

from sampling import sample_tasks


def fixture(operations,sources=30):
    return [{'id':f'{source:03}_{operation}','source_id':f'{source:03}',
             'operation':operation,'instruction':'Original instruction'}
            for source in range(sources) for operation in operations]


class SamplingTests(unittest.TestCase):
    def test_asset20_has_five_equal_operations_and_twenty_source_designs(self):
        rows=fixture(['delete','insert','replace','move','scale'])
        picked=sample_tasks(rows,20)
        self.assertEqual(Counter(x['operation'] for x in picked),dict.fromkeys(['delete','insert','replace','move','scale'],4))
        self.assertEqual(len({x['source_id'] for x in picked}),20)

    def test_text20_covers_six_operations_with_balanced_quotas(self):
        picked=sample_tasks(fixture(['delete','insert','replace','move','scale','color']),20)
        counts=Counter(x['operation'] for x in picked)
        self.assertEqual(len(counts),6);self.assertEqual(sorted(counts.values()),[3,3,3,3,4,4])
        self.assertEqual(len({x['source_id'] for x in picked}),20)

    def test_matching_recovers_unique_sources_when_greedy_assignment_could_fail(self):
        rows=[{'id':'a1','source_id':'one','operation':'a'},
              {'id':'a2','source_id':'two','operation':'a'},
              {'id':'b1','source_id':'one','operation':'b'}]
        for seed in range(10):
            picked=sample_tasks(rows,2,seed)
            self.assertEqual({x['id'] for x in picked},{'a2','b1'})

    def test_input_order_and_task_contents_do_not_change_reproducibility(self):
        rows=fixture(['delete','insert','replace','move','scale']);before=deepcopy(rows)
        self.assertEqual(sample_tasks(rows,20),sample_tasks(list(reversed(rows)),20))
        self.assertEqual(rows,before)
        self.assertNotEqual(sample_tasks(rows,20,42),sample_tasks(rows,20,43))

    def test_unavoidable_repeated_sources_are_balanced_and_tasks_stay_distinct(self):
        rows=fixture(['delete','insert','replace','move','scale'],3)
        picked=sample_tasks(rows,12)
        self.assertEqual(len({x['id'] for x in picked}),12)
        self.assertEqual(Counter(x['source_id'] for x in picked),{'000':4,'001':4,'002':4})

    def test_small_operation_capacity_is_respected(self):
        rows=fixture(['a'],1)+fixture(['b'],8)
        picked=sample_tasks(rows,5)
        self.assertEqual(Counter(x['operation'] for x in picked),{'a':1,'b':4})
        self.assertEqual(len({x['source_id'] for x in picked}),5)

    def test_full_selection_keeps_every_original_task(self):
        rows=fixture(['a','b','c'],5)
        self.assertEqual({x['id'] for x in sample_tasks(rows,len(rows))},{x['id'] for x in rows})

    def test_invalid_selection_is_rejected(self):
        rows=fixture(['a','b'])
        for count in (0,-1,True,1.5,len(rows)+1):
            with self.subTest(count=count),self.assertRaises(ValueError):sample_tasks(rows,count)
        with self.assertRaises(ValueError):sample_tasks(rows,2,True)
        with self.assertRaises(ValueError):sample_tasks(rows+[rows[0]],2)


if __name__=='__main__':unittest.main()
