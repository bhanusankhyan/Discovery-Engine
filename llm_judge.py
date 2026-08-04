from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# 1. Define the Custom G-Eval Metric
coherence_metric = GEval(
    name="Coherence",
    # The LLM will use these steps to generate a prompt and evaluate
    evaluation_steps=[
        "Evaluate whether the response flows smoothly.",
        "Check if the logical sequencing makes sense.",
        "Ensure there are no contradictory statements within the output."
    ],
    # Define what the metric looks at (in this case, just the generated output)
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    # You can specify the model to use as the judge (e.g., GPT-4 is recommended)
    model="gpt-4o"
)

# 2. Define the Test Case
# This represents one run of your application
test_case = LLMTestCase(
    input="what the relation between quick comerce and e commerce",
    actual_output="""The Relationship Between E-commerce and Quick Commerce
E-commerce and quick commerce are two interconnected segments of the retail landscape, each catering to different consumer needs and preferences. While e-commerce encompasses a broad range of online shopping experiences, quick commerce focuses specifically on the rapid delivery of goods, particularly fast-moving consumer goods (FMCGs) that cater to impulsive consumer behavior.

E-commerce Overview
E-commerce refers to any transaction that occurs online, including retail, services, and more. It typically involves longer delivery timelines, ranging from one to several days, which aligns with planned purchases. E-commerce platforms like Amazon and Flipkart offer a wide variety of products, appealing to consumers who are willing to wait for delivery in exchange for a broader selection and often better pricing. The growth of e-commerce has been driven by factors such as rising digital adoption and increased consumer affluence, transforming purchasing behaviors significantly.

Quick Commerce Dynamics
Quick commerce, on the other hand, is a specialized subset of e-commerce that emphasizes speed and convenience, often promising delivery within minutes. This model thrives in urban environments where immediate delivery needs are heightened, capitalizing on dense populations that prefer quick shopping experiences. Companies like Blinkit and Swiggy Instamart exemplify how quick commerce can complement traditional e-commerce by catering to impulsive buyer profiles who favor FMCGs, thus driving increased sales velocity in the e-grocery segment.

Complementary Roles
The relationship between e-commerce and quick commerce is characterized by their complementary roles. Quick commerce captures urgent, impulse-driven purchases, while traditional e-commerce caters to more considered buying decisions. For instance, quick commerce accounted for about two-thirds of all online grocery orders in India in 2024, indicating its growing importance in the market. As consumer expectations shift towards immediacy and convenience, quick commerce platforms are increasingly integrating with traditional e-commerce models to enhance consumer satisfaction and logistical efficiency.

Technological Integration
Both sectors benefit from advancements in technology, which streamline the shopping experience and provide deeper insights into consumer preferences. Quick commerce platforms utilize dark stores and micro-fulfillment centers to optimize delivery times, while e-commerce platforms leverage robust inventory management systems to handle a wider variety of products. This technological synergy allows both sectors to adapt to changing consumer demands effectively.

Conclusion
In summary, the relationship between e-commerce and quick commerce is one of interdependence, where each model serves distinct consumer needs while also enhancing the overall retail ecosystem. As quick commerce continues to grow, driven by consumer demand for speed and convenience, it is likely to further influence the strategies of traditional e-commerce platforms, leading to a more integrated and responsive market environment. Understanding this relationship is crucial for businesses aiming to navigate the rapidly evolving retail landscape effectively."""
)

# 3. Execute the Evaluation
coherence_metric.measure(test_case)

# 4. Access the Results
print(f"Score: {coherence_metric.score}") # e.g., 0.6
print(f"Reasoning: {coherence_metric.reason}") # The Chain-of-Thought explanation from the judge
