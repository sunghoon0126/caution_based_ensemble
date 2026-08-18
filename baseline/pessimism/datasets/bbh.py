# pessimism/datasets/bbh.py
import logging
import re
import random
from baseline.pessimism.datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from typing import Dict, Any, Optional, List

# Import the new base classes
from baseline.pessimism.datasets.base_problem import SpecialCompletionProblem, SpecialCompletionDataset

logger = logging.getLogger(__name__)

# Constants for parsing logic
ANSWER_TRIGGER = "####"
INVALID_ANS = "[invalid]"

# BBH has 27 subsets - list them all
BBH_SUBSETS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting"
]

# Random-guess baselines per subset (fractions in [0,1]) from the BBH paper.
# Where the paper reports category averages (e.g., Logical Deduction, Tracking Shuffled Objects),
# we apply that value to each corresponding subset.
BBH_RANDOM_BASELINES: Dict[str, float] = {
    "boolean_expressions": 0.50,
    "causal_judgement": 0.50,
    "date_understanding": 0.172,
    "disambiguation_qa": 0.332,
    "dyck_languages": 0.012,
    "formal_fallacies": 0.25,
    "geometric_shapes": 0.116,
    "hyperbaton": 0.50,
    # Logical Deduction (avg)
    "logical_deduction_three_objects": 0.225,
    "logical_deduction_five_objects": 0.225,
    "logical_deduction_seven_objects": 0.225,
    "movie_recommendation": 0.25,
    "multistep_arithmetic_two": 0.0,
    "navigate": 0.50,
    "object_counting": 0.0,
    "penguins_in_a_table": 0.0,
    "reasoning_about_colored_objects": 0.119,
    "ruin_names": 0.25,
    "salient_translation_error_detection": 0.167,
    "snarks": 0.50,
    "sports_understanding": 0.50,
    "temporal_sequences": 0.25,
    # Tracking Shuffled Objects (avg)
    "tracking_shuffled_objects_three_objects": 0.225,
    "tracking_shuffled_objects_five_objects": 0.225,
    "tracking_shuffled_objects_seven_objects": 0.225,
    "web_of_lies": 0.50,
    "word_sorting": 0.0,
}

def _infer_num_options(question_text: str) -> Optional[int]:
    """Best-effort inference of number of options in a BBH question.

    Tries common patterns like (A)/(B)/..., 'A.', 'B.', 'A)', 'B)'. If two or more
    are detected, returns that count. Falls back to binary heuristics if keywords
    like 'true/false' or 'yes/no' are present. Returns None if unknown.
    """
    if not isinstance(question_text, str) or not question_text:
        return None

    text = question_text

    # Normalize spacing for easier regex matching
    # Look for lettered options A-E (sometimes more) with common delimiters
    import re as _re
    letter_pattern = _re.compile(r"(?im)(?:^|\s)[\[(]?[A-Ja-j][\])\.:]")
    letters = letter_pattern.findall(text)
    # Count unique ordered labels by scanning for A, B, C, ... occurrences
    if letters:
        # Estimate by counting the maximum contiguous sequence starting from A
        # as tasks usually enumerate A, B, C, D
        # Fallback: count distinct option markers in the text
        option_markers = _re.findall(r"(?im)[\[(]?([A-Ja-j])[\])\.:]", text)
        if option_markers:
            unique_letters = set([m.upper() for m in option_markers])
            if len(unique_letters) >= 2:
                return len(unique_letters)

    # Numeric options like 1., 2., 3., 4.
    numeric_pattern = _re.compile(r"(?im)(?:^|\s)(?:\d{1,2})[\.)]\s")
    numeric = numeric_pattern.findall(text)
    if numeric:
        # Conservative guess: count distinct numeric indices seen up to 10
        nums = _re.findall(r"(?im)(?:^|\s)(\d{1,2})[\.)]\s", text)
        if nums:
            try:
                unique_nums = set(int(n) for n in nums if int(n) <= 20)
                if len(unique_nums) >= 2:
                    return len(unique_nums)
            except Exception:
                pass

    # Binary heuristics
    lowered = text.lower()
    if ("true" in lowered and "false" in lowered) or ("yes" in lowered and "no" in lowered):
        return 2

    return None

def _infer_random_baseline(question_text: str) -> Optional[float]:
    """Return random-guess baseline accuracy for a single question as a float in [0,1].

    If the number of options can be inferred (n>=2), returns 1/n. If binary
    heuristics trigger, returns 0.5. Otherwise returns None.
    """
    n_opts = _infer_num_options(question_text)
    if n_opts is not None and n_opts >= 2:
        try:
            return 1.0 / float(n_opts)
        except Exception:
            return None
    return None

def extract_bbh_answer(response: str) -> str:
    """
    Parses the models response to extract the final answer for BBH.
    Looks for #### trigger and returns the answer after it.
    """
    if not response:
        return INVALID_ANS
    response = response.strip()
    response = response.replace(ANSWER_TRIGGER, f" {ANSWER_TRIGGER} ")
    parts = response.split(ANSWER_TRIGGER)
    answer_flag = len(parts) > 1
    if answer_flag:
        potential_answer_part = parts[-1]
    else:
        potential_answer_part = parts[-1]
    
    # Take first line after ####, strip and clean
    potential_answer_part = potential_answer_part.strip()
    lines = potential_answer_part.split('\n')
    if lines and lines[0].strip():
        answer = lines[0].strip()
        # Remove common punctuation and extra spaces
        answer = re.sub(r'[.!?]+$', '', answer)
        return answer.strip()
    
    return INVALID_ANS

class BBHProblem(SpecialCompletionProblem):
    """
    Represents a single BBH problem, inheriting from SpecialCompletionProblem.
    Implements the specific correctness checking logic for BBH.
    """
    def __init__(
        self,
        problem: str, # Question text
        answer: str,  # Ground truth answer as a string
        parent_uuid: Optional[str] = None,
        generation_config: Optional[Dict] = None,
        extra: Optional[dict] = None,
    ):
        # Clean and normalize the ground truth answer
        self.clean_answer = self.clean_ground_truth(answer)
        super().__init__(problem, answer, parent_uuid, generation_config, extra)
        # Expose subset as a direct attribute as well for convenience
        try:
            self.subset = (extra or {}).get("bbh_subset")
        except Exception:
            self.subset = None

    def clean_ground_truth(self, answer_str: str) -> str:
        """Cleans the ground truth answer string from the dataset."""
        if not answer_str:
            return ""
        
        # Remove common formatting and normalize
        cleaned = answer_str.strip()
        cleaned = re.sub(r'[.!?]+$', '', cleaned)
        return cleaned.strip().lower()

    def check_correctness(self, response: str) -> bool:
        """
        Parses the models's response, extracts the answer,
        and compares it against the cleaned ground truth answer using strip/lowercase comparison.
        """
        extracted_answer = extract_bbh_answer(response)
        if extracted_answer == INVALID_ANS:
            return False
        
        # Normalize extracted answer for comparison
        extracted_cleaned = extracted_answer.strip().lower()
        
        # Strip and lowercase comparison as requested
        is_correct = (extracted_cleaned == self.clean_answer)
        return is_correct

    def _get_hashing_components(self) -> List[Any]:
        """Return components relevant for hashing BBHProblem."""
        return super()._get_hashing_components()

class BBHDataset(SpecialCompletionDataset):
    """
    Dataset handler for Big-Bench Hard, loading data and creating BBHProblem instances.
    Loads all 27 subsets and creates a unified dataset.
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
        shuffle_before_limit: bool = True,
        **kwargs
    ):
        # Initialize the base class
        super().__init__(seed=seed, **kwargs)

        self.name_or_path = "lighteval/big_bench_hard" if name_or_path is None else name_or_path
        self.config_name = config_name
        self.seed = seed
        self.limit_problems = limit_problems
        self.shuffle_before_limit = shuffle_before_limit

        # Few-shot example loading
        self.fewshot_examples = []
        if fewshot_num > 0:
            if not fewshot_split:
                logger.warning("fewshot_num > 0 but fewshot_split is not specified. Cannot load few-shot examples.")
            else:
                logger.info(f"Loading raw few-shot data from split '{fewshot_split}' for selection...")
                try:
                    # Load all BBH subsets for few-shot
                    raw_fewshot_datasets = []
                    for subset in BBH_SUBSETS:
                        try:
                            # Check available splits first
                            try:
                                subset_dataset = load_dataset(
                                    self.name_or_path, subset, split=fewshot_split
                                )
                            except ValueError as e:
                                if "Unknown split" in str(e):
                                    # Try the opposite split as fallback
                                    fallback_split = "train" if fewshot_split == "test" else "test"
                                    logger.info(f"Split '{fewshot_split}' not available for subset '{subset}', trying '{fallback_split}'")
                                    subset_dataset = load_dataset(
                                        self.name_or_path, subset, split=fallback_split
                                    )
                                else:
                                    raise e
                            
                            # Add subset info to each example
                            subset_with_info = subset_dataset.map(lambda x: {**x, "subset": subset})
                            raw_fewshot_datasets.append(subset_with_info)
                        except Exception as e:
                            logger.warning(f"Failed to load subset {subset} for few-shot: {e}")
                    
                    if raw_fewshot_datasets:
                        # Concatenate all subset datasets
                        raw_fewshot_dataset = concatenate_datasets(raw_fewshot_datasets)
                        
                        # Use the base class method to select raw examples
                        super().select_fewshot_examples(raw_fewshot_dataset, fewshot_num, seed=self.seed)

                        # Parse the selected raw examples into BBHProblem objects
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
                        logger.warning("No BBH subsets could be loaded for few-shot examples.")
                        self.fewshot_examples = []

                except Exception as e:
                    logger.error(f"Failed to load or select few-shot examples from split '{fewshot_split}': {e}", exc_info=True)
                    self.fewshot_examples = []

        # Main dataset loading - load all BBH subsets
        logger.info(f"Loading BBH main dataset: split='{split}', loading all {len(BBH_SUBSETS)} subsets")
        try:
            # Load all BBH subsets
            hf_main_datasets = []
            for subset in BBH_SUBSETS:
                try:
                    # Check available splits first
                    try:
                        subset_dataset = load_dataset(self.name_or_path, subset, split=split)
                    except ValueError as e:
                        if "Unknown split" in str(e) and split == "test":
                            # Try train split as fallback
                            logger.info(f"Split '{split}' not available for subset '{subset}', trying 'train'")
                            subset_dataset = load_dataset(self.name_or_path, subset, split="train")
                        else:
                            raise e
                    
                    # Add subset info to each example
                    subset_with_info = subset_dataset.map(lambda x: {**x, "subset": subset})
                    hf_main_datasets.append(subset_with_info)
                    logger.info(f"Loaded {len(subset_dataset)} examples from subset '{subset}'")
                except Exception as e:
                    logger.warning(f"Failed to load subset {subset}: {e}")
            
            if not hf_main_datasets:
                raise ValueError("Could not load any BBH subsets")
            
            # Concatenate all subset datasets
            hf_main_dataset = concatenate_datasets(hf_main_datasets)
            logger.info(f"Loaded total of {len(hf_main_dataset)} main instances from all BBH subsets.")
            
        except Exception as e:
            logger.error(f"Failed to load main dataset split '{split}': {e}", exc_info=True)
            raise

        # Parse main dataset & generate prompts
        self.parse_hf_dataset(hf_main_dataset)
        self.generate_prompt_text()

        # Build helper maps: uuid -> subset
        self.uuid2subset = {}
        for p in getattr(self, 'problems', []) or []:
            try:
                self.uuid2subset[p.uuid] = getattr(p, 'subset', None) or (getattr(p, 'extra', {}) or {}).get('bbh_subset')
            except Exception:
                pass

        # Populate random baselines per subset using paper values; fallback to heuristic if absent
        self.random_baseline_by_subset: Dict[str, Optional[float]] = {}
        for subset in BBH_SUBSETS:
            if subset in BBH_RANDOM_BASELINES:
                self.random_baseline_by_subset[subset] = BBH_RANDOM_BASELINES[subset]
            else:
                # Fallback: try to infer from questions of this subset if paper value missing
                subset_vals: List[float] = []
                for p in getattr(self, 'problems', []) or []:
                    try:
                        sub = getattr(p, 'subset', None) or (getattr(p, 'extra', {}) or {}).get('bbh_subset')
                        if sub == subset:
                            bl = _infer_random_baseline(getattr(p, 'problem', None))
                            if bl is not None:
                                subset_vals.append(float(bl))
                    except Exception:
                        pass
                self.random_baseline_by_subset[subset] = (sum(subset_vals) / len(subset_vals)) if subset_vals else None

        # Log a brief summary for visibility
        known = {k: v for k, v in self.random_baseline_by_subset.items() if v is not None}
        if known:
            logger.info(f"Loaded random baselines for {len(known)}/{len(BBH_SUBSETS)} BBH subsets (paper or inferred). Examples: {list(known.items())[:3]}")

    def parse_data_instance(self, data: Dict[str, Any], extra: Optional[Dict] = None) -> BBHProblem:
        """Parses a raw data instance (dict) from Hugging Face dataset into a BBHProblem."""
        if extra is None: extra = {}
        
        # BBH dataset format - check available fields
        if "input" in data and "target" in data:
            # Standard BBH format
            question = data.get("input")
            answer_str = data.get("target")
        elif "question" in data and "answer" in data:
            # Alternative format
            question = data.get("question")
            answer_str = data.get("answer")
        else:
            # Try to infer from available keys
            available_keys = list(data.keys())
            logger.warning(f"Unknown BBH data format. Available keys: {available_keys}")
            # Use first available text field as question
            text_fields = [k for k in available_keys if isinstance(data.get(k), str) and len(data.get(k, "")) > 10]
            if len(text_fields) >= 2:
                question = data[text_fields[0]]
                answer_str = data[text_fields[1]]
            else:
                raise ValueError(f"Cannot parse BBH data format. Available keys: {available_keys}")

        if question is None or answer_str is None:
            raise ValueError(f"Missing question or answer in raw data instance: {data}")

        # Add subset information to extra
        if "subset" in data:
            extra["bbh_subset"] = data["subset"]

        # Define generation config for BBH
        generation_config = {"stop_sequences": ["\n\n", "Q:", "Question:"]}

        return BBHProblem(
            problem=question,
            answer=answer_str,
            extra=extra,
            generation_config=generation_config,
        )

    def parse_hf_dataset(self, hf_dataset: HFDataset):
        """Iterates through the provided Hugging Face dataset and populates self.problems."""
        self.problems = []
        if hf_dataset is None:
             logger.error("No Hugging Face dataset provided to parse_hf_dataset.")
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

        # Apply shuffling and limiting
        if self.shuffle_before_limit and len(self.problems) > 0:
            logger.info(f"Shuffling {len(self.problems)} problems with seed {self.seed}")
            # Set random seed for reproducible shuffling
            random.seed(self.seed)
            random.shuffle(self.problems)
            
        if self.limit_problems is not None:
            self.problems = self.problems[:self.limit_problems]
            logger.info(f"Limited dataset to {len(self.problems)} problems")
            
        logger.info(f"Successfully parsed {len(self.problems)} main BBH problems.")