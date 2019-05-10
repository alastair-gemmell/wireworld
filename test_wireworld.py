import unittest
import wireworld as ww
import wireworld.engine as engine
import numpy as np
import os


class TestStateEvolution(unittest.TestCase):
    
    def setUp(self):
        self.ww_eng = engine.WireworldEngine()
        self.test_array = np.array([[1,1,1],
                                    [3,3,3],
                                    [3,3,3],
                                    [1,2,0]])
         
        self.next_state = np.array([[2,2,2],
                                    [1,3,1],
                                    [1,1,3],
                                    [2,3,0]])
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)    
    
    def test_next_value_empty_to_empty(self):
        self.assertEqual(self.ww_eng.get_next_value((3, 2), self.test_array), ww.EMPTY, "expected EMPTY (0)")

    def test_next_value_head_to_tail(self):
        self.assertEqual(self.ww_eng.get_next_value((3, 0), self.test_array), ww.TAIL, "expected TAIL (2)")

    def test_next_value_conductor_to_head_one_head_adjacent(self):
        self.assertEqual(self.ww_eng.get_next_value((2, 0), self.test_array), ww.HEAD, "expected HEAD (1)")

    def test_next_value_conductor_to_head_two_heads_adjacent(self):
        self.assertEqual(self.ww_eng.get_next_value((1, 0), self.test_array), ww.HEAD, "expected HEAD (1)")

    def test_next_value_conductor_to_conductor_zero_heads_adjacent(self):
        self.assertEqual(self.ww_eng.get_next_value((2, 2), self.test_array), ww.CONDUCTOR, "expected CONDUCTOR (3)")

    def test_next_value_conductor_to_conductor_three_heads_adjacent(self):
        self.assertEqual(self.ww_eng.get_next_value((1, 1), self.test_array), ww.CONDUCTOR, "expected CONDUCTOR (3)")

    def test_next_state(self):
        computed_next_state = self.ww_eng.get_next_state(self.test_array)
        reference_next_state = self.next_state
        msg = "Computed next state does not match reference next state!"
        self.assertTrue(np.array_equal(computed_next_state, reference_next_state), msg)


class TestDisplay(unittest.TestCase):
    def setUp(self):
        self.ww_eng = engine.WireworldEngine() 
        self.state = np.array([[0,1],
                               [2,3]])
                                            
        self.display_array = np.array([[ww.BLACK, ww.BLACK, ww.BLUE, ww.BLUE],
                                       [ww.BLACK, ww.BLACK, ww.BLUE, ww.BLUE],
                                       [ww.RED, ww.RED, ww.YELLOW, ww.YELLOW],
                                       [ww.RED, ww.RED, ww.YELLOW, ww.YELLOW]])  
        self.scale_factor = 2
    
    def tearDown(self):
        unittest.TestCase.tearDown(self)    
    
    def test_display_array_from_state(self):
        computed_display_array = self.ww_eng.get_display_array_from_state(self.state, self.scale_factor)
        reference_display_array = self.display_array
        msg = "Computed display array does not match reference display array!"
        self.assertTrue(np.array_equal(computed_display_array, reference_display_array), msg)
        
        
class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.ww_eng = engine.WireworldEngine()
        self.state = np.array([[0,1],
                               [2,3]]) 
        self.test_file = 'states/test_save.py'

    def tearDown(self):
        unittest.TestCase.tearDown(self)   
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
    def test_load_and_save(self):
        self.ww_eng.save_state(self.state, self.test_file)
        loaded_state = self.ww_eng.load_state(self.test_file)
        msg = "Saved and re-loaded state does not match original state"
        self.assertTrue(np.array_equal(loaded_state, self.state), msg)

     
if __name__ == '__main__':
    unittest.main()
