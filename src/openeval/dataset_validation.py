"""Dataset validation and quality assessment utilities."""

from typing import List, Optional, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass, asdict

from .core import Dataset, Example


@dataclass
class DatasetQualityReport:
    """Report of dataset quality assessment."""
    total_examples: int
    valid_examples: int
    invalid_examples: int
    avg_input_length: float
    avg_reference_length: float
    unique_inputs: int
    unique_references: int
    duplicate_pairs: int
    empty_inputs: int
    empty_references: int
    encoding_issues: int
    format_issues: List[str]
    quality_score: float
    recommendations: List[str]


class DatasetValidator:
    """Validates datasets and assesses quality."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise errors on validation failures. If False, collect issues.
        """
        self.strict = strict
    
    def validate_example(self, example: Example) -> Tuple[bool, List[str]]:
        """
        Validate a single example.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not hasattr(example, 'id') or not example.id:
            issues.append("Missing or empty example ID")
        
        if not hasattr(example, 'input'):
            issues.append("Missing input field")
        elif example.input is None:
            issues.append("Input is None")
        elif isinstance(example.input, str) and not example.input.strip():
            issues.append("Empty input string")
        
        if not hasattr(example, 'reference'):
            issues.append("Missing reference field") 
        elif example.reference is None:
            issues.append("Reference is None")
        elif isinstance(example.reference, str) and not example.reference.strip():
            issues.append("Empty reference string")
        
        # Check for encoding issues
        try:
            if isinstance(example.input, str):
                example.input.encode('utf-8')
            if isinstance(example.reference, str):
                example.reference.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Unicode encoding issues")
        
        # Check for suspicious patterns
        if isinstance(example.input, str) and len(example.input) > 10000:
            issues.append("Unusually long input (>10k chars)")
        
        if isinstance(example.reference, str) and len(example.reference) > 5000:
            issues.append("Unusually long reference (>5k chars)")
        
        # Check for HTML/XML artifacts
        if isinstance(example.input, str):
            if re.search(r'<[^>]+>', example.input):
                issues.append("Possible HTML/XML tags in input")
        
        if isinstance(example.reference, str):
            if re.search(r'<[^>]+>', example.reference):
                issues.append("Possible HTML/XML tags in reference")
        
        return len(issues) == 0, issues
    
    def assess_quality(self, dataset: Dataset) -> DatasetQualityReport:
        """
        Perform comprehensive quality assessment of a dataset.
        
        Args:
            dataset: Dataset to assess
            
        Returns:
            Quality report with metrics and recommendations
        """
        examples = list(dataset)
        total = len(examples)
        
        if total == 0:
            return DatasetQualityReport(
                total_examples=0,
                valid_examples=0,
                invalid_examples=0,
                avg_input_length=0,
                avg_reference_length=0,
                unique_inputs=0,
                unique_references=0,
                duplicate_pairs=0,
                empty_inputs=0,
                empty_references=0,
                encoding_issues=0,
                format_issues=["Dataset is empty"],
                quality_score=0.0,
                recommendations=["Add examples to the dataset"]
            )
        
        valid_count = 0
        input_lengths = []
        reference_lengths = []
        inputs = []
        references = []
        input_ref_pairs = []
        empty_inputs = 0
        empty_references = 0
        encoding_issues = 0
        format_issues = []
        
        for i, example in enumerate(examples):
            is_valid, issues = self.validate_example(example)
            
            if is_valid:
                valid_count += 1
            else:
                format_issues.extend([f"Example {i}: {issue}" for issue in issues])
                if "Unicode encoding issues" in str(issues):
                    encoding_issues += 1
            
            # Collect statistics
            try:
                input_str = str(example.input) if example.input is not None else ""
                ref_str = str(example.reference) if example.reference is not None else ""
                
                if not input_str.strip():
                    empty_inputs += 1
                else:
                    input_lengths.append(len(input_str))
                    inputs.append(input_str)
                
                if not ref_str.strip():
                    empty_references += 1
                else:
                    reference_lengths.append(len(ref_str))
                    references.append(ref_str)
                
                input_ref_pairs.append((input_str, ref_str))
                
            except Exception:
                format_issues.append(f"Example {i}: Error processing strings")
        
        # Calculate statistics
        avg_input_len = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        avg_ref_len = sum(reference_lengths) / len(reference_lengths) if reference_lengths else 0
        
        unique_inputs = len(set(inputs)) if inputs else 0
        unique_references = len(set(references)) if references else 0
        unique_pairs = len(set(input_ref_pairs)) if input_ref_pairs else 0
        duplicate_pairs = total - unique_pairs
        
        # Calculate quality score (0-1)
        validity_score = valid_count / total
        uniqueness_score = unique_pairs / total if total > 0 else 0
        completeness_score = 1.0 - (empty_inputs + empty_references) / (2 * total) if total > 0 else 0
        encoding_score = 1.0 - encoding_issues / total if total > 0 else 1.0
        
        quality_score = (validity_score + uniqueness_score + completeness_score + encoding_score) / 4
        
        # Generate recommendations
        recommendations = []
        if validity_score < 0.9:
            recommendations.append("Fix validation issues in examples")
        if uniqueness_score < 0.8:
            recommendations.append("Remove duplicate input-reference pairs")
        if empty_inputs > 0:
            recommendations.append(f"Fill {empty_inputs} empty input fields")
        if empty_references > 0:
            recommendations.append(f"Fill {empty_references} empty reference fields")
        if encoding_issues > 0:
            recommendations.append("Fix Unicode encoding issues")
        if avg_input_len < 10:
            recommendations.append("Input examples seem very short - consider more detailed inputs")
        if avg_input_len > 1000:
            recommendations.append("Input examples are very long - consider truncation")
        if quality_score > 0.8:
            recommendations.append("Dataset quality is good!")
        
        return DatasetQualityReport(
            total_examples=total,
            valid_examples=valid_count,
            invalid_examples=total - valid_count,
            avg_input_length=avg_input_len,
            avg_reference_length=avg_ref_len,
            unique_inputs=unique_inputs,
            unique_references=unique_references,
            duplicate_pairs=duplicate_pairs,
            empty_inputs=empty_inputs,
            empty_references=empty_references,
            encoding_issues=encoding_issues,
            format_issues=format_issues[:20],  # Limit to first 20 issues
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def validate_dataset(self, dataset: Dataset, save_report: Optional[Path] = None) -> bool:
        """
        Validate entire dataset and optionally save report.
        
        Args:
            dataset: Dataset to validate
            save_report: Path to save JSON report (optional)
            
        Returns:
            True if dataset passes validation
        """
        report = self.assess_quality(dataset)
        
        if save_report:
            save_report.parent.mkdir(parents=True, exist_ok=True)
            with open(save_report, 'w') as f:
                json.dump(asdict(report), f, indent=2)
        
        # Consider dataset valid if quality score > 0.7 and no critical issues
        is_valid = (
            report.quality_score > 0.7 and
            report.total_examples > 0 and
            report.valid_examples > 0
        )
        
        if self.strict and not is_valid:
            raise ValueError(f"Dataset validation failed. Quality score: {report.quality_score:.2f}")
        
        return is_valid


def validate_jsonl_file(file_path: Path, required_fields: Optional[List[str]] = None) -> DatasetQualityReport:
    """
    Validate a JSONL file format for dataset use.
    
    Args:
        file_path: Path to JSONL file
        required_fields: List of required field names (default: ['input', 'reference'])
        
    Returns:
        Quality report
    """
    if required_fields is None:
        required_fields = ['input', 'reference']
    
    validator = DatasetValidator()
    
    try:
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Create example from JSON data
                    example_id = data.get('id', f"line_{line_num}")
                    input_data = data.get('input', data.get('question', ''))
                    reference_data = data.get('reference', data.get('answer', data.get('output', '')))
                    meta_data = {k: v for k, v in data.items() if k not in ['id', 'input', 'reference']}
                    
                    example = Example(
                        id=example_id,
                        input=input_data,
                        reference=reference_data,
                        meta=meta_data or None
                    )
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    # Create a dummy example to report the error
                    example = Example(
                        id=f"line_{line_num}",
                        input=f"JSON_PARSE_ERROR: {str(e)}",
                        reference="",
                        meta=None
                    )
                    examples.append(example)
        
        # Create a temporary dataset
        class TempDataset(Dataset):
            name = str(file_path)
            def __init__(self, examples):
                self._examples = examples
            def __iter__(self):
                return iter(self._examples)
        
        dataset = TempDataset(examples)
        return validator.assess_quality(dataset)
        
    except Exception as e:
        return DatasetQualityReport(
            total_examples=0,
            valid_examples=0,
            invalid_examples=0,
            avg_input_length=0,
            avg_reference_length=0,
            unique_inputs=0,
            unique_references=0,
            duplicate_pairs=0,
            empty_inputs=0,
            empty_references=0,
            encoding_issues=0,
            format_issues=[f"Failed to read file: {str(e)}"],
            quality_score=0.0,
            recommendations=[f"Fix file reading issue: {str(e)}"]
        )
