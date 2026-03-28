from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = """
Machine learning is a branch of artificial intelligence that allows systems to learn 
and improve from experience without being explicitly programmed. It focuses on developing 
computer programs that can access data and use it to learn for themselves. The process 
begins with observations or data, such as examples, direct experience, or instruction, 
to look for patterns in data and make better decisions in the future.
"""

result = summarizer(text, max_length=50, min_length=20, do_sample=False)
print(result[0]['summary_text'])