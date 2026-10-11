import tempfile
import unittest
from pathlib import Path
import numpy as np
from korean_rag.runtime import parent_window, decode_span, verify_bundle, write_manifest, PortableRAG


class RuntimeTests(unittest.TestCase):
    def test_reader_window_follows_retrieval_and_preserves_source(self):
        text='abcdefghij'
        offsets=[(i,i+1) for i in range(len(text))]
        self.assertEqual(parent_window(text,offsets,7,9,4),('ghij',6,10))
        self.assertEqual(parent_window(text,offsets,0,1,4),('abcd',0,4))

    def test_span_excludes_question_tokens_and_rejects_null(self):
        offsets=[(0,0),(0,2),(0,1),(1,2),(0,0)]
        logits=np.array([0,20,2,3,0])
        result=decode_span('AB',offsets,[None,0,1,1,None],logits,logits,0)
        self.assertEqual(result['answer'],'B')
        logits[0]=40
        self.assertEqual(decode_span('AB',offsets,[None,0,1,1,None],logits,logits,0)['answer'],'')

    def test_manifest_missing_required_and_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            (path/'config.json').write_text('{}')
            write_manifest(path,{})
            with self.assertRaises(ValueError):
                verify_bundle(path)
            (path/'manifest.json').write_text('{"schema_version":1,"files":{"../outside":{"bytes":0,"sha256":"x"}}}')
            with self.assertRaises(ValueError):
                verify_bundle(path)

    def test_threshold_and_source_selection(self):
        c={'answer':'1999년','context':'1999년 기록','rrf_score':.03,'reranker_score':.8,'reader_score':.4,'rank':1,
           'title':'t','chunk_id':0,'context_id':'d','reader_window_start':0,'answer_char_start':0,'answer_char_end':5}
        thresholds={'min_rrf_score':0,'min_reranker_score':.5,'min_reader_score':.1}
        self.assertTrue(PortableRAG.select_answer('몇 년?',[],[c],thresholds,{})['answered'])
        self.assertFalse(PortableRAG.select_answer('2000년 기록?',[],[c],thresholds,{})['answered'])


if __name__=='__main__':
    unittest.main()
