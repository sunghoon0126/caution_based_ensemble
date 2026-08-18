# pessimism/datasets/mathgsm.py
import logging
import re
import pandas as pd
from baseline.pessimism.datasets import load_dataset, Dataset as HFDataset
from typing import Dict, Any, Optional, List
import requests
import io

# Import the new base classes
from baseline.pessimism.datasets.base_problem import SpecialCompletionProblem, SpecialCompletionDataset

# Official MATH dataset checking logic
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the models output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if verbose:
        print(f'Checking <{str1}> vs <{str2}>')
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

logger = logging.getLogger(__name__)

# Constants for parsing logic - using GSM8K format
ANSWER_TRIGGER = "####"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

INVALID_ANS = "[invalid]"

def extract_mathgsm_answer(response: str) -> str:
    """
    Parses the models response to extract the final answer for MATH-GSM problems.
    Uses GSM8K-style #### format but handles mathematical expressions.
    """
    if not response:
        return INVALID_ANS
    
    response = response.strip()
    
    # First try to find #### pattern
    if ANSWER_TRIGGER in response:
        response = response.replace(ANSWER_TRIGGER, f" {ANSWER_TRIGGER} ")
        parts = response.split(ANSWER_TRIGGER)
        if len(parts) > 1:
            potential_answer_part = parts[-1].strip()
            # Take the first line after ####
            lines = potential_answer_part.split('\n')
            if lines and lines[0].strip():
                return lines[0].strip()
    
    # Fallback: Look for common answer patterns if no #### found
    # Look for \boxed{...} pattern (common in math)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for "Answer:" pattern  
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for final numerical answer or expression at the end
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('Step') and not line.startswith('Therefore'):
            # Try to extract mathematical content
            # Look for standalone numbers, fractions, expressions
            if re.match(r'^[\d\.\-\+\*/\(\)\\\{\}\^_\$a-zA-Z\s]+$', line) and len(line) < 50:
                return line
    
    return INVALID_ANS

class MathGSMProblem(SpecialCompletionProblem):
    """
    Represents a single MATH problem with GSM8K-style answer format.
    Uses #### parsing but handles mathematical expressions.
    """
    def __init__(
        self,
        problem: str,  # Question text
        solution: str, # Solution text
        answer: str,   # Ground truth answer as a string (should be final answer only)
        parent_uuid: Optional[str] = None,
        generation_config: Optional[Dict] = None,
        extra: Optional[dict] = None,
    ):
        # Use the answer directly (CSV format gives us clean final answers)
        self.clean_answer = answer
        print(f"Answer: <{self.clean_answer}>")
        super().__init__(problem, solution, parent_uuid, generation_config, extra)

    def check_correctness(self, response: str) -> bool:
        """
        Parses the models's response using GSM8K-style #### format,
        extracts the answer, and compares it against the ground truth.
        """
        extracted_answer = extract_mathgsm_answer(response)
        if extracted_answer == INVALID_ANS:
            return False
        
        # Use official MATH dataset equivalence checking
        is_correct = is_equiv(extracted_answer, self.clean_answer, verbose=True)
        return is_correct

    def _get_hashing_components(self) -> List[Any]:
        """Return components relevant for hashing MathGSMProblem."""
        return super()._get_hashing_components()

class MathGSMDataset(SpecialCompletionDataset):
    """
    Dataset handler for MATH with GSM8K-style answer format.
    Loads MATH data but expects #### answer format.
    """
    def __init__(
        self,
        seed: int = 1,
        split: str = "test",
        name_or_path: Optional[str] = None,
        config_name: Optional[str] = None,
        fewshot_split: Optional[str] = "train",
        fewshot_num: int = 0,
        limit_problems: Optional[int] = None,
        math_split: str = "math_test",  # "math_test" or "math_500_test"
        **kwargs
    ):
        # Initialize the base class
        super().__init__(seed=seed, **kwargs)

        self.name_or_path = "hendrycks/competition_math" if name_or_path is None else name_or_path
        self.config_name = config_name
        self.seed = seed
        self.limit_problems = limit_problems
        self.math_split = math_split

        # Few-shot example loading
        self.fewshot_examples = []
        if fewshot_num > 0:
            if not fewshot_split:
                logger.warning("fewshot_num > 0 but fewshot_split is not specified. Cannot load few-shot examples.")
            else:
                logger.info(f"Loading raw few-shot data from split '{fewshot_split}' for selection...")
                try:
                    # Try CSV first for clean Answer field, then fallback to HuggingFace
                    if fewshot_split == "train":
                        # For train split, try loading a training CSV if available
                        raw_fewshot_dataset = self._load_from_csv(split_override="math_train")
                        if raw_fewshot_dataset is None:
                            raw_fewshot_dataset = self._load_hf_dataset(fewshot_split)
                    else:
                        raw_fewshot_dataset = self._load_hf_dataset(fewshot_split)
                    
                    if raw_fewshot_dataset:
                        super().select_fewshot_examples(raw_fewshot_dataset, fewshot_num, seed=self.seed)
                        
                        # Parse selected examples
                        parsed_fewshot_examples = []
                        for i, raw_example in enumerate(self.fewshot_examples):
                            try:
                                parsed_fewshot_examples.append(
                                    self.parse_data_instance(raw_example, extra={"origin": f"fewshot_{fewshot_split}", "fewshot_original_index": i})
                                )
                            except ValueError as e:
                                logger.warning(f"Skipping selected few-shot example {i} due to parsing error: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected error parsing selected few-shot example {i}: {e}", exc_info=True)

                        self.fewshot_examples = parsed_fewshot_examples
                        logger.info(f"Successfully selected and parsed {len(self.fewshot_examples)} few-shot examples.")
                    else:
                        logger.warning("Could not load few-shot data from Hugging Face, skipping few-shot examples.")
                        self.fewshot_examples = []

                except Exception as e:
                    logger.error(f"Failed to load or select few-shot examples from split '{fewshot_split}': {e}", exc_info=True)
                    self.fewshot_examples = []

        # Main dataset loading
        logger.info(f"Loading MATH-GSM main dataset: split='{split}', math_split='{math_split}'")
        try:
            # For MATH-500, prefer HuggingFace dataset which has clean fields
            if self.math_split == "math_500_test":
                hf_main_dataset = self._load_hf_dataset(split)
                if hf_main_dataset is None:
                    logger.warning("HuggingFace MATH-500 loading failed, trying CSV as fallback")
                    hf_main_dataset = self._load_from_csv()
            else:
                # For other splits, try CSV first (gives clean "Answer" field)
                hf_main_dataset = self._load_from_csv()
                if hf_main_dataset is None:
                    logger.warning("CSV loading failed, trying Hugging Face as fallback")
                    hf_main_dataset = self._load_hf_dataset(split)
            
            if hf_main_dataset:
                logger.info(f"Loaded {len(hf_main_dataset)} main instances from split '{split}'.")
            else:
                raise ValueError("Could not load MATH dataset from any source")
                
        except Exception as e:
            logger.error(f"Failed to load main dataset split '{split}': {e}", exc_info=True)
            raise

        # Parse main dataset & generate prompts
        self.parse_hf_dataset(hf_main_dataset)
        self.generate_prompt_text()

    def _load_hf_dataset(self, split: str) -> Optional[HFDataset]:
        """Try to load from Hugging Face datasets."""
        try:
            # For MATH-500, use the clean HuggingFaceH4 dataset
            if self.math_split == "math_500_test":
                dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
                return dataset
            elif self.name_or_path == "hendrycks/competition_math":
                dataset = load_dataset(self.name_or_path, split=split)
                return dataset
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            return None

    def _load_from_csv(self, split_override: Optional[str] = None) -> Optional[List[Dict]]:
        """Load from OpenAI CSV files as fallback."""
        try:
            # Use split_override if provided, otherwise use self.math_split
            csv_split = split_override if split_override else self.math_split
            csv_url = f"XXXX{csv_split}.csv"
            logger.info(f"Loading MATH dataset from CSV: {csv_url}")
            
            response = requests.get(csv_url)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            # Convert to list of dicts - use CSV field names directly
            data_list = []
            for _, row in df.iterrows():
                data_list.append({
                    "Question": row.get("Question", ""),
                    "Answer": row.get("Answer", "")  # Use Answer field directly
                })
            
            return data_list
        except Exception as e:
            logger.error(f"Failed to load from CSV: {e}")
            return None

    def parse_data_instance(self, data: Dict[str, Any], extra: Optional[Dict] = None) -> MathGSMProblem:
        """Parses a raw data instance (dict) into a MathGSMProblem."""
        if extra is None:
            extra = {}
        
        # Handle different data formats
        if "problem" in data and "answer" in data:
            # HuggingFace MATH-500 format (preferred)
            question = data["problem"]
            answer = data["answer"]  # Use clean answer field directly
            solution = data["solution"]
        elif "problem" in data and "solution" in data:
            # Original HuggingFace format (fallback)
            question = data["problem"]
            answer = data["solution"]
            solution = data["solution"]
        elif "Question" in data and "Answer" in data:
            # CSV format
            question = data["Question"]
            answer = data["Answer"]
            solution = data["Answer"]
        else:
            raise ValueError(f"Missing required fields in raw data instance: {data}")

        if question is None or answer is None:
            raise ValueError(f"Missing 'problem'/'Question' or 'answer'/'Answer'/'solution' in raw data instance: {data}")

        # Define generation config - GSM8K style
        generation_config = {"stop_sequences": ["\n\n", "Q:", "Question:"]}

        return MathGSMProblem(
            problem=question,
            solution=solution,
            answer=answer,
            extra=extra,
            generation_config=generation_config,
        )

    def parse_hf_dataset(self, hf_dataset):
        """Iterates through the provided dataset and populates self.problems."""
        self.problems = []
        if hf_dataset is None:
            logger.error("No dataset provided to parse_hf_dataset.")
            return

        logger.info(f"Parsing {len(hf_dataset)} main dataset instances...")
        for i, instance_data in enumerate(hf_dataset):
            try:
                problem_instance = self.parse_data_instance(instance_data, extra={"original_index": i, "origin": "main"})
                self.problems.append(problem_instance)
            except ValueError as e:
                logger.warning(f"Skipping main instance {i} due to parsing error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing main instance {i}: {e}", exc_info=True)

        if self.limit_problems is not None:
            self.problems = self.problems[:self.limit_problems]
            
        logger.info(f"Successfully parsed {len(self.problems)} main MATH-GSM problems.") 