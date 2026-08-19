# pessimism/datasets/gsm8k.py
import logging
import re
from datasets import load_dataset, Dataset as HFDataset # Import Hugging Face Dataset type
from typing import Dict, Any, Optional, List

# Import the new base classes
from baseline.pessimism.datasets.base_problem import SpecialCompletionProblem, SpecialCompletionDataset

logger = logging.getLogger(__name__)

# Constants for parsing logic
ANSWER_TRIGGER = "####"
INVALID_ANS = "[invalid]"

# --- Helper Function: extract_gsm8k_answer (keep as before) ---
def extract_gsm8k_answer(response: str) -> str:
    """
    Parses the models response to extract the final numerical answer for GSM8K.
    Based on the logic provided in the prompt.
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
    potential_answer_part = potential_answer_part.replace(",", "").strip()
    numbers = re.findall(r'-?\d+\.?\d*', potential_answer_part)
    if not numbers:
        return INVALID_ANS
    if answer_flag:
        extracted_num_str = numbers[0]
    else:
        extracted_num_str = numbers[-1]
    if extracted_num_str.endswith("."):
        extracted_num_str = extracted_num_str[:-1]
    try:
        float(extracted_num_str)
        return extracted_num_str
    except ValueError:
        logger.warning(f"Extracted '{extracted_num_str}' is not a valid number representation.")
        return INVALID_ANS

# --- Class: GSM8KProblem (keep as before) ---
class GSM8KProblem(SpecialCompletionProblem):
    """
    Represents a single GSM8K problem, inheriting from SpecialCompletionProblem.
    Implements the specific correctness checking logic for GSM8K.
    """
    def __init__(
        self,
        problem: str, # Question text
        answer: str,  # Ground truth numerical answer as a string
        parent_uuid: Optional[str] = None,
        generation_config: Optional[Dict] = None,
        extra: Optional[dict] = None,
    ):
        cleaned_answer = extract_gsm8k_answer(self.clean_ground_truth(answer))
        self.clean_answer = cleaned_answer
        super().__init__(problem, answer, parent_uuid, generation_config, extra)

    def clean_ground_truth(self, answer_str: str) -> str:
        """Cleans the ground truth answer string from the dataset."""
        parts = answer_str.split(ANSWER_TRIGGER)
        if len(parts) > 1:
            gt_part = parts[-1]
        else:
            gt_part = answer_str
        gt_part = gt_part.replace(",", "").strip()
        numbers = re.findall(r'-?\d+\.?\d*', gt_part)
        if numbers:
             num_str = numbers[0]
             if num_str.endswith("."):
                 num_str = num_str[:-1]
             try:
                 float(num_str)
                 return num_str
             except ValueError:
                 pass
        logger.warning(f"Could not extract clean numerical ground truth from: '{answer_str}'. Using raw split part: '{gt_part.strip()}'")
        return gt_part.strip()

    def check_correctness(self, response: str) -> bool:
        """
        Parses the models's response, extracts the numerical answer,
        and compares it against the cleaned ground truth answer.
        """
        extracted_answer = extract_gsm8k_answer(response)
        if extracted_answer == INVALID_ANS:
            return False
        # print(f'<<<{extracted_answer}>>> [[[{self.clean_answer}]]]')
        is_correct = (extracted_answer == self.clean_answer)
        return is_correct

    # --- ADDED Method ---
    def _get_hashing_components(self) -> List[Any]:
        """Return components relevant for hashing GSM8KProblem."""
        # Since GSM8KProblem doesn't add new fields needing hashing beyond
        # what SpecialCompletionProblem (and BaseProblem) already cover,
        # we can just call the superclass implementation.
        return super()._get_hashing_components()

# --- Class: GSM8KDataset (Corrected __init__ using base class method) ---
class GSM8KDataset(SpecialCompletionDataset):
    """
    Dataset handler for GSM8K, loading data and creating GSM8KProblem instances.
    Inherits from SpecialCompletionDataset and handles few-shot examples using the base class method.
    """
    def __init__(
        self,
        seed: int = 1,
        split: str = "test",
        name_or_path: Optional[str] = None,
        config_name: Optional[str] = None,
        fewshot_split: Optional[str] = "train", # Default few-shot split to 'train'
        fewshot_num: int = 0,
        limit_problems: Optional[int] = None,
        **kwargs
    ):
        # Initialize the base class (SpecialCompletionDataset)
        # Pass seed and other relevant kwargs like tokenizer_name_or_path, system_prompt etc.
        super().__init__(seed=seed, **kwargs)

        self.name_or_path = "gsm8k" if name_or_path is None else name_or_path
        self.config_name = "main" if config_name is None else config_name
        self.seed = seed # Store seed if needed elsewhere
        self.limit_problems = limit_problems

        # --- Few-shot Example Loading & Selection ---
        self.fewshot_examples = [] # Ensure initialized
        if fewshot_num > 0:
            if not fewshot_split:
                logger.warning("fewshot_num > 0 but fewshot_split is not specified. Cannot load few-shot examples.")
            else:
                logger.info(f"Loading raw few-shot data from split '{fewshot_split}' for selection...")
                try:
                    # Load the *entire* raw few-shot dataset split
                    raw_fewshot_dataset: HFDataset = load_dataset(
                        self.name_or_path, self.config_name, split=fewshot_split
                    )

                    # Use the base class method to select raw examples
                    # Note: Base class select_fewshot_examples expects a BaseDataset-like object or list
                    # Passing the raw Hugging Face dataset directly works because it's list-like (indexable and has __len__)
                    super().select_fewshot_examples(raw_fewshot_dataset, fewshot_num, seed=self.seed)
                    # At this point, self.fewshot_examples contains the *raw selected data* (dicts)

                    # Now, parse the selected raw examples into GSM8KProblem objects
                    parsed_fewshot_examples = []
                    for i, raw_example in enumerate(self.fewshot_examples): # Iterate over the raw data selected by base class
                        try:
                            parsed_fewshot_examples.append(
                                self.parse_data_instance(raw_example, extra={"origin": f"fewshot_{fewshot_split}", "fewshot_original_index": i})
                            )
                        except ValueError as e:
                            logger.warning(f"Skipping selected few-shot example {i} due to parsing error: {e}")
                        except Exception as e:
                            logger.error(f"Unexpected error parsing selected few-shot example {i}: {e}", exc_info=True)

                    self.fewshot_examples = parsed_fewshot_examples # Replace raw data with parsed problems
                    logger.info(f"Successfully selected and parsed {len(self.fewshot_examples)} few-shot examples.")

                except Exception as e:
                    logger.error(f"Failed to load or select few-shot examples from split '{fewshot_split}': {e}", exc_info=True)
                    self.fewshot_examples = [] # Ensure it's empty on error


        # --- Main Dataset Loading ---
        logger.info(f"Loading GSM8K main dataset: split='{split}'")
        try:
            # Store the loaded dataset in a temporary variable or directly pass to parse_hf_dataset
            hf_main_dataset = load_dataset(self.name_or_path, self.config_name, split=split)
            logger.info(f"Loaded {len(hf_main_dataset)} main instances from split '{split}'.")
        except Exception as e:
            logger.error(f"Failed to load main dataset split '{split}': {e}", exc_info=True)
            raise

        # --- Parse Main Dataset & Generate Prompts ---
        self.parse_hf_dataset(hf_main_dataset) # Pass the loaded dataset to parse
        self.generate_prompt_text() # Generates prompts using self.problems and the *parsed* self.fewshot_examples

    def parse_data_instance(self, data: Dict[str, Any], extra: Optional[Dict] = None) -> GSM8KProblem:
        """Parses a raw data instance (dict) from Hugging Face dataset into a GSM8KProblem."""
        if extra is None: extra = {}
        question = data.get("question")
        answer_str = data.get("answer")
        if question is None or answer_str is None:
            raise ValueError(f"Missing 'question' or 'answer' in raw data instance: {data}")

        # Define generation configs (e.g., stop sequences) if needed for GSM8K
        generation_config = {"stop_sequences": ["\n\n", "Q:", "Question:"]} # Example

        return GSM8KProblem(
            problem=question,
            answer=answer_str, # Pass the raw answer string, cleaning happens in GSM8KProblem.__init__
            extra=extra,
            generation_config=generation_config,
        )

    def parse_hf_dataset(self, hf_dataset: HFDataset):
        """Iterates through the provided Hugging Face dataset and populates self.problems."""
        self.problems = [] # Clear existing problems
        if hf_dataset is None:
             logger.error("No Hugging Face dataset provided to parse_hf_dataset.")
             return

        logger.info(f"Parsing {len(hf_dataset)} main dataset instances...")
        for i, instance_data in enumerate(hf_dataset):
            try:
                # Add origin info to main problems as well
                problem_instance = self.parse_data_instance(instance_data, extra={"original_index": i, "origin": "main"})
                self.problems.append(problem_instance)
            except ValueError as e:
                 logger.warning(f"Skipping main instance {i} due to parsing error: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error parsing main instance {i}: {e}", exc_info=True)

        if self.limit_problems is not None:
            self.problems = self.problems[:self.limit_problems]
            
        logger.info(f"Successfully parsed {len(self.problems)} main GSM8K problems.")

    # generate_prompt_text is inherited from BaseDataset via SpecialCompletionDataset
    # _get_prompt_generation_kwargs is inherited from SpecialCompletionDataset
    # select_fewshot_examples is inherited BUT we use it internally in __init__ on raw data
