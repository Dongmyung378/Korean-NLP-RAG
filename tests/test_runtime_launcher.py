"""The public launcher selects CPU by default and loads models once per session."""
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

from scripts.launch_rag import main


class RuntimeLauncherTests(unittest.TestCase):
    def run_launcher(self, arguments):
        with patch('scripts.launch_rag.sys.argv',['launch_rag.py',*arguments]), \
             patch('scripts.launch_rag.setup',return_value=Path('private-python')) as setup, \
             patch('scripts.launch_rag.verify_bundle',return_value={'valid':True}), \
             patch('scripts.launch_rag.subprocess.run',return_value=Mock(returncode=0)) as run:
            result=main()
            return result,setup.call_args,run.call_args

    def test_default_cpu_persistent_session(self):
        result,setup,run=self.run_launcher(['--no-download'])
        self.assertEqual(result,0)
        command=run.args[0]
        self.assertIn('--interactive',command)
        self.assertEqual(command[command.index('--device')+1],'cpu')
        self.assertEqual(setup.args[2],'cpu')

    def test_cuda_requires_explicit_opt_in(self):
        result,setup,run=self.run_launcher(['--no-download','--device','cuda','--question','서울의 인구는?'])
        self.assertEqual(result,0)
        self.assertEqual(setup.args[2],'cuda')
        self.assertNotIn('--interactive',run.args[0])

    def test_questions_file_runs_batch_without_interactive_loop(self):
        result,setup,run=self.run_launcher(['--no-download','--questions-file','questions.txt'])
        self.assertEqual(result,0)
        self.assertIn('--questions-file',run.args[0])
        self.assertNotIn('--interactive',run.args[0])


if __name__=='__main__':
    unittest.main()
