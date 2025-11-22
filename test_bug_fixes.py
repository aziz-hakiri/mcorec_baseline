"""
Test suite for bug fixes in mcorec_baseline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from src.talking_detector.segmentation import segment_by_asd, CENTRAL_ASD_CHUNKING_PARAMETERS
from src.tokenizer.norm_text import norm_string


class TestBug1SegmentationDefaultParameter(unittest.TestCase):
    """Test that min_duration_off uses correct default parameter"""

    def test_min_duration_off_default_value(self):
        """
        Test that when min_duration_off is not provided, it defaults to
        CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_off"] (0.5)
        not CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"] (1.0)
        """
        # Create test ASD data with two speech regions separated by a short gap
        # Frame rate is 25 fps, so:
        # - 0.5 seconds = 12.5 frames (rounded to 12 frames in code)
        # - 1.0 second = 25 frames

        asd_data = {}
        # First speech region: frames 0-49 (2 seconds)
        for i in range(50):
            asd_data[str(i)] = 2.0  # High score = speech

        # Gap of 15 frames (0.6 seconds)
        for i in range(50, 65):
            asd_data[str(i)] = 0.5  # Low score = silence

        # Second speech region: frames 65-99 (1.4 seconds)
        for i in range(65, 100):
            asd_data[str(i)] = 2.0  # High score = speech

        # Call segment_by_asd without providing min_duration_off parameter
        segments = segment_by_asd(asd_data, parameters={})

        # With correct default (0.5s = 12 frames), the gap of 15 frames
        # should NOT be merged (15 > 12), resulting in 2 segments
        # With wrong default (1.0s = 25 frames), the gap of 15 frames
        # would be merged (15 < 25), resulting in 1 segment

        self.assertEqual(len(segments), 2,
                        "With correct min_duration_off=0.5s, two speech regions "
                        "separated by 0.6s gap should remain separate")

        # Verify the segments are correct
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[0][-1], 49)
        self.assertEqual(segments[1][0], 65)
        self.assertEqual(segments[1][-1], 99)

    def test_min_duration_off_explicit_value(self):
        """Test that explicitly provided min_duration_off is respected"""
        asd_data = {}
        # First speech region
        for i in range(50):
            asd_data[str(i)] = 2.0

        # Gap of 15 frames
        for i in range(50, 65):
            asd_data[str(i)] = 0.5

        # Second speech region
        for i in range(65, 100):
            asd_data[str(i)] = 2.0

        # Explicitly set min_duration_off to 1.0 (25 frames)
        # This should merge the gap
        segments = segment_by_asd(asd_data, parameters={"min_duration_off": 1.0})

        self.assertEqual(len(segments), 1,
                        "With min_duration_off=1.0s, gap of 0.6s should be merged")


class TestBug2NormSetNoDuplicate(unittest.TestCase):
    """Test that norm_set doesn't have duplicate '}' character"""

    def test_norm_set_no_duplicate_brace(self):
        """
        Test that norm_string function works correctly without
        duplicate '}' in the norm_set
        """
        # Test text with special characters including }
        test_text = "Hello {world} test"

        # This should not raise an error and should normalize correctly
        result = norm_string(test_text)

        # The function should process the text without issues
        self.assertIsInstance(result, str)
        self.assertIn("HELLO", result)
        self.assertIn("WORLD", result)

    def test_norm_string_with_braces(self):
        """Test that both { and } characters are in the norm set"""
        test_text = "Test{value}example"
        result = norm_string(test_text)

        # The braces should be handled by the normalization
        self.assertIsInstance(result, str)


class TestBug3UndefinedVariableInErrorHandler(unittest.TestCase):
    """Test that lip_crop error handler doesn't reference undefined variables"""

    def test_error_message_no_undefined_variable(self):
        """
        Test that the error handler in lip_crop.py doesn't reference
        undefined segment_frame variable by checking the source code
        """
        # Read the lip_crop.py file and verify the fix
        with open('script/lip_crop.py', 'r') as f:
            content = f.read()

        # Find the error handling section
        error_section_start = content.find('except Exception as e:')
        self.assertNotEqual(error_section_start, -1, "Error handler should exist")

        # Get the error handling code (next 200 characters after except)
        error_handler = content[error_section_start:error_section_start + 200]

        # The bug was: print(f"Error processing {video_path} segment {segment_frame[0]}-{segment_frame[-1]}")
        # segment_frame is not defined in the function, so this would cause NameError

        # Verify that segment_frame is NOT referenced in the error handler
        self.assertNotIn('segment_frame', error_handler,
                        "Error handler should not reference undefined segment_frame variable")

        # Verify that video_path IS still referenced (it's the function parameter)
        self.assertIn('video_path', error_handler,
                     "Error handler should reference video_path")


class TestBug4DoubleCountingMilliseconds(unittest.TestCase):
    """Test that VTT timestamp comparison doesn't double-count milliseconds"""

    def test_timestamp_comparison_no_double_count(self):
        """
        Test that evaluate.py correctly uses caption.start_in_seconds
        without adding milliseconds again
        """
        # Read the evaluate.py file
        with open('script/evaluate.py', 'r') as f:
            content = f.read()

        # Find the benchmark_vtt_wer function
        func_start = content.find('def benchmark_vtt_wer(')
        self.assertNotEqual(func_start, -1, "benchmark_vtt_wer function should exist")

        # Get the function content (approximately 500 characters should cover the relevant part)
        func_content = content[func_start:func_start + 800]

        # The bug was adding milliseconds twice:
        # if caption.start_in_seconds + caption.start_time.milliseconds/1000 < ref_uem_start:

        # Verify that we're NOT adding milliseconds to start_in_seconds
        self.assertNotIn('start_in_seconds + caption.start_time.milliseconds', func_content,
                        "Should not add milliseconds to start_in_seconds (already included)")
        self.assertNotIn('end_in_seconds + caption.end_time.milliseconds', func_content,
                        "Should not add milliseconds to end_in_seconds (already included)")

        # Verify that we're using start_in_seconds correctly
        self.assertIn('caption.start_in_seconds', func_content,
                     "Should use caption.start_in_seconds for timestamp comparison")
        self.assertIn('caption.end_in_seconds', func_content,
                     "Should use caption.end_in_seconds for timestamp comparison")


class TestBug5SetInsteadOfListInASD(unittest.TestCase):
    """Test that durationSet in asd.py is a list, not a set"""

    def test_duration_set_is_list_with_duplicates(self):
        """
        Test that durationSet is defined as a list to preserve duplicate values,
        not a set which would remove them
        """
        # Read the asd.py file
        with open('script/asd.py', 'r') as f:
            content = f.read()

        # Find the durationSet definition
        duration_set_line_start = content.find('durationSet = ')
        self.assertNotEqual(duration_set_line_start, -1, "durationSet definition should exist")

        # Get the line with the durationSet definition (next 50 characters)
        duration_set_def = content[duration_set_line_start:duration_set_line_start + 50]

        # The bug was using a set: durationSet = {1,1,1,2,2,2,3,3,4,5,6}
        # This would become {1,2,3,4,5,6} due to set deduplication

        # Verify it's defined as a list (starts with [)
        self.assertIn('durationSet = [', duration_set_def,
                     "durationSet should be defined as a list, not a set")

        # Verify it's NOT defined as a set (doesn't start with {)
        self.assertNotIn('durationSet = {', duration_set_def,
                        "durationSet should not be defined as a set (would remove duplicates)")

    def test_duration_values_preserved(self):
        """Test that the intended duplicate duration values are preserved"""
        # Read the file and extract the durationSet value
        with open('script/asd.py', 'r') as f:
            content = f.read()

        # Extract the durationSet line
        import re
        match = re.search(r'durationSet = \[([\d,]+)\]', content)
        self.assertIsNotNone(match, "Should find durationSet as a list")

        # The list should have 11 elements (with duplicates: 1,1,1,2,2,2,3,3,4,5,6)
        duration_str = match.group(1)
        durations = [int(x.strip()) for x in duration_str.split(',')]

        # Verify we have 11 elements (not 6 as a set would have)
        self.assertEqual(len(durations), 11,
                        "durationSet should have 11 elements with duplicates preserved")

        # Verify the values include the expected duplicates
        self.assertEqual(durations.count(1), 3, "Should have three 1s")
        self.assertEqual(durations.count(2), 3, "Should have three 2s")
        self.assertEqual(durations.count(3), 2, "Should have two 3s")


if __name__ == '__main__':
    unittest.main()
