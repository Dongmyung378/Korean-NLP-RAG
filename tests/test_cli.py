import io
import unittest
from unittest.mock import patch
from korean_rag.__main__ import main

class CLITests(unittest.TestCase):
    def test_persistent_mode_loads_once_and_survives_invalid_question(self):
        output=io.StringIO()
        with patch('korean_rag.runtime.PortableRAG') as factory, patch('builtins.input',side_effect=['A question?', 'x'*1001, 'Another question?', '/quit']), patch('sys.stdout',output), patch('sys.stderr',io.StringIO()):
            factory.return_value.answer.return_value={'answer':'answer','answer_source':'extractive_reader'}
            self.assertEqual(main(['--interactive','--json']),0)
            self.assertEqual(factory.call_count,1)
            self.assertEqual(factory.return_value.answer.call_count,2)
        self.assertEqual(len(output.getvalue().splitlines()),3)
        self.assertIn('invalid_input',output.getvalue())

if __name__=='__main__':
    unittest.main()
