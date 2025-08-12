"""Tests for prompt templating system."""

import pytest

from openeval.prompt import PromptTemplate, build_prompt, COMMON_TEMPLATES


class TestPromptTemplate:
    """Test PromptTemplate functionality."""

    def test_format_template(self):
        """Test basic format string templates."""
        template = PromptTemplate("Question: {question}\nAnswer:")
        result = template.render(question="What is 2+2?")
        assert result == "Question: What is 2+2?\nAnswer:"
    
    def test_auto_detect_format(self):
        """Test automatic detection of format templates."""
        template = PromptTemplate("Question: {question}")
        assert template.template_type == "format"
    
    def test_auto_detect_jinja2(self):
        """Test automatic detection of Jinja2 templates."""
        template = PromptTemplate("Question: {{question}}")
        assert template.template_type == "jinja2"
    
    def test_jinja2_template(self):
        """Test Jinja2 template rendering."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate("Question: {{question.strip()}}\nAnswer:")
        result = template.render(question="  What is 2+2?  ")
        assert result == "Question: What is 2+2?\nAnswer:"
    
    def test_jinja2_conditional(self):
        """Test Jinja2 conditional logic."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate("{% if name %}Hello {{name}}{% else %}Hello World{% endif %}")
        
        result1 = template.render(name="Alice")
        assert result1 == "Hello Alice"
        
        result2 = template.render()
        assert result2 == "Hello World"
    
    def test_jinja2_loop(self):
        """Test Jinja2 loop functionality."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate("""{% for item in items %}
{{loop.index}}. {{item}}
{% endfor %}""")
        
        result = template.render(items=["apple", "banana", "cherry"])
        expected = "1. apple\n2. banana\n3. cherry\n"
        assert result == expected
    
    def test_get_variables_format(self):
        """Test variable extraction from format templates."""
        template = PromptTemplate("Hello {name}, you are {age} years old")
        variables = template.get_variables()
        assert set(variables) == {"name", "age"}
    
    def test_get_variables_jinja2(self):
        """Test variable extraction from Jinja2 templates."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate("Hello {{name}}, you are {{age}} years old")
        variables = template.get_variables()
        assert set(variables) == {"name", "age"}
    
    def test_alpha_filter(self):
        """Test custom alpha filter for multiple choice."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate("""{% for choice in choices %}
{{loop.index0 | alpha}}. {{choice}}
{% endfor %}""")
        
        result = template.render(choices=["Apple", "Banana", "Cherry"])
        expected = "A. Apple\nB. Banana\nC. Cherry\n"
        assert result == expected


class TestBuildPrompt:
    """Test build_prompt helper function."""
    
    def test_build_prompt_string(self):
        """Test building prompt from template string."""
        result = build_prompt(
            "Question: {question}\nAnswer:",
            {"question": "What is 2+2?"}
        )
        assert result == "Question: What is 2+2?\nAnswer:"
    
    def test_build_prompt_template_object(self):
        """Test building prompt from PromptTemplate object."""
        template = PromptTemplate("Question: {question}\nAnswer:")
        result = build_prompt(template, {"question": "What is 2+2?"})
        assert result == "Question: What is 2+2?\nAnswer:"


class TestCommonTemplates:
    """Test common template patterns."""
    
    def test_multiple_choice_template(self):
        """Test multiple choice template."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate(COMMON_TEMPLATES["multiple_choice"])
        result = template.render(
            question="What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"]
        )
        
        expected = """What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid
Answer:"""
        assert result == expected
    
    def test_qa_template(self):
        """Test QA template."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate(COMMON_TEMPLATES["qa"])
        result = template.render(question="What is 2+2?")
        
        expected = "Question: What is 2+2?\nAnswer:"
        assert result == expected
    
    def test_few_shot_template(self):
        """Test few-shot template."""
        pytest.importorskip("jinja2")
        
        template = PromptTemplate(COMMON_TEMPLATES["few_shot"])
        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "3+3", "output": "6"}
        ]
        result = template.render(examples=examples, input="4+4")
        
        expected = """2+2
4

3+3
6

4+4"""
        assert result == expected


def test_missing_jinja2():
    """Test graceful handling when jinja2 is not available."""
    # This would need to be tested in an environment without jinja2
    # For now, we just test that the error message is appropriate
    try:
        template = PromptTemplate("{{test}}", template_type="jinja2")
        # If jinja2 is available, this should work
        result = template.render(test="value")
        assert result == "value"
    except ImportError as e:
        assert "jinja2 is required" in str(e)
