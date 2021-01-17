import unittest
import sys
sys.path.append(".")
from src.preprocess import get_lemma,remove_punctuation,clean_sentence

class TestPreprocess(unittest.TestCase):

    def test_lemma(self):
        self.assertEqual(get_lemma("feet"), "foot", "Should be feet")

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("hello world!?, Yes this is our 1code. @adel"), ["hello", "world", "Yes", "this", "is", "our" ,"1code","adel"], "No punctuation")


    def test_clean_sentence(self):
          self.assertEqual(clean_sentence("hello worlds!?, Yes this is our 1code. @adel"), ["hello", "world","yes","code"], "Should be clean")

if __name__ == '__main__':
    unittest.main()