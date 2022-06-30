
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from TDA.TDA import main

from arguments import args

class TestTDA:
    def test_add(self):
        ret = main()
        assert(ret==0)



if __main__ == "__main__":
    b = TestTDA()
