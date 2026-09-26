import os
import time
import dotenv
from groq import Groq
from log import Logger


class Summarizer:
    def __init__(self, env_file="key.env"):
        """
        Initializes the Summarizer class and configures the Groq API client.

        Args:
            env_file (str, optional): The environment file containing the API keys. Default is "key.env".
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=False)
        self.model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.low_model = os.getenv("GROQ_LOW_MODEL_NAME", "openai/gpt-oss-20b")
        self.client = Groq()

    def claim_title_summarize(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Generates a concise, highly searchable query for the given claim using the Groq API.

        Args:
            text (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens for the completion. Default is 1024.
            temperature (float, optional): Controls randomness. Default is 0.5.
            stop (str or None, optional): Optional stop sequence. Default is None.

        Returns:
            tuple: (summary_string, total_tokens_used)
        """
        self.logger.info("Starting claim title summarization.")
        self.logger.debug("Input text: %s...", text[:200])

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI designed to rephrase a claim into a concise, specific, and highly searchable query. Focus on preserving all critical details such as names, dates, locations, or key terms, but avoid unnecessary words. Provide only the text without any additional formatting.",
                    },
                    {"role": "user", "content": text},
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
            )

            summary = response.choices[0].message.content
            if not summary:
                return None, 0

            summary = summary.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            self.logger.info("Generated scraping summary: %s...", summary[:100])
            return summary, tokens

        except Exception as e:
            self.logger.error("Error generating summary: %s", e)
            return None, 0

    def claim_summary_summarize(
        self, text, max_tokens=1024, temperature=0.0, stop=None
    ):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's claim as one concise factual statement "
                            "in English. Preserve the original meaning and all important "
                            "details. Do not answer, verify, confirm, or refute the claim. "
                            "Do not add information that is not present in the input. "
                            "If the input is a question, rewrite it as a neutral statement "
                            "describing what needs to be verified. "
                            "Return only the rewritten claim."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                model=self.low_model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
            )

            summary = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            if not summary:
                return None, 0

            return summary.strip(), tokens

        except Exception as e:
            self.logger.error("Error generating claim summary: %s", e)
            return None, 0
        
    def generate_summary(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Generates a general summary for the given text.

        Args:
            text (str): The text to be summarized.
            max_tokens (int): Maximum number of tokens for the summary.
            temperature (float): Sampling temperature.
            stop (str or list): Stop sequence(s) for the model.

        Returns:
            tuple: (summary_string, total_tokens_used)
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a summarizer, be specific. Don't use lists or bullet points. Provide only the string without specifying that it is a summary. Translate in English.",
                    },
                    {"role": "user", "content": text},
                ],
                model=self.low_model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop,
            )

            summary = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            if not summary:
                return None, 0
            return summary.strip(), tokens

        except Exception as e:
            self.logger.error("Error generating text summary: %s", e)
            return None, 0

    def summarize_texts(
        self,
        texts,
        max_tokens=1024,
        temperature=0.5,
        stop=None,
        token_cut=20000,
        sleep_temperature=0.0003,
    ):
        """
        Generates summaries for a list of texts and returns the token usage.

        Args:
            texts (list): List of strings to summarize.
            max_tokens (int, optional): Maximum number of tokens for each completion. Default is 1024.
            temperature (float, optional): Controls randomness. Default is 0.5.
            stop (str or None, optional): Optional stop sequence. Default is None.
            token_cut (int, optional): Maximum characters to pass to the model.
            sleep_temperature (float, optional): Multiplier to calculate sleep time to avoid rate limits.

        Returns:
            tuple: (list_of_summaries, total_tokens_used)
        """
        self.logger.info("Starting batch summarization for %d texts.", len(texts))
        summaries = []
        total_tokens_used = 0

        for index, text in enumerate(texts):
            self.logger.info("Summarizing text %d/%d...", index + 1, len(texts))
            cutted_text = text[:token_cut]

            try:
                summary, tokens = self.generate_summary(
                    text=cutted_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )

                if summary:
                    summaries.append(summary)
                    total_tokens_used += tokens

                    sleep_time = len(cutted_text) * sleep_temperature
                    time.sleep(sleep_time)
                else:
                    self.logger.warning("No summary returned for text %d.", index + 1)
                    summaries.append(None)

            except Exception as e:
                self.logger.error("Error summarizing text %d: %s", index + 1, e)
                summaries.append(None)

        self.logger.info(
            "Batch summarization completed. Total tokens: %d", total_tokens_used
        )

        return summaries, total_tokens_used
