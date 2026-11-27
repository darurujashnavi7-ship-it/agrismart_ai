import os
import unittest
from unittest.mock import Mock,patch

import requests

from backend import chatbot


class ChatbotTests(unittest.TestCase):
    def test_configured_chatbot_uses_groq_and_conversation_history(self):
        response=Mock()
        response.raise_for_status.return_value=None
        response.json.return_value={
            "choices":[{"message":{"content":"Check soil moisture before watering."}}]
        }

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response) as post:
                result=chatbot.answer(
                    "Should I irrigate today?",
                    history=[("user","I grow tomatoes."),("assistant","What soil do you have?")],
                )

        self.assertEqual(result,"Check soil moisture before watering.")
        payload=post.call_args.kwargs["json"]
        self.assertEqual(payload["model"],chatbot.MODEL)
        self.assertEqual(payload["messages"][-3]["role"],"user")
        self.assertEqual(payload["messages"][-3]["content"],"I grow tomatoes.")

    def test_selected_language_is_included_in_the_response_instructions(self):
        response=Mock()
        response.raise_for_status.return_value=None
        response.json.return_value={
            "choices":[{"message":{"content":"मिट्टी की नमी जाँचें।"}}]
        }

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response) as post:
                result=chatbot.answer("Should I irrigate today?",language="Hindi")

        self.assertEqual(result,"मिट्टी की नमी जाँचें।")
        self.assertIn("Respond in Hindi",post.call_args.kwargs["json"]["messages"][0]["content"])

    def test_voice_transcription_uses_selected_language_and_returns_text(self):
        response=Mock()
        response.raise_for_status.return_value=None
        response.json.return_value={"text":"ನನ್ನ ಬೆಳೆ ಹಳದಿ ಆಗುತ್ತಿದೆ"}

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response) as post:
                result=chatbot.transcribe_audio(b"wav-data",language="Kannada")

        self.assertEqual(result,"ನನ್ನ ಬೆಳೆ ಹಳದಿ ಆಗುತ್ತಿದೆ")
        self.assertEqual(post.call_args.kwargs["data"]["language"],"kn")
        self.assertEqual(post.call_args.kwargs["data"]["model"],chatbot.TRANSCRIPTION_MODEL)

    def test_voice_transcription_requires_api_key(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            with self.assertRaisesRegex(chatbot.ChatbotError,"GROQ_API_KEY"):
                chatbot.transcribe_audio(b"wav-data")

    def test_voice_transcription_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError,"supported AgriBot languages"):
            chatbot.transcribe_audio(b"wav-data",language="Klingon")

    def test_chatbot_adds_soil_context_and_safety_constraints(self):
        response=Mock()
        response.raise_for_status.return_value=None
        response.json.return_value={
            "choices":[{"message":{"content":"The entered nitrogen class is low."}}]
        }

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response) as post:
                chatbot.answer(
                    "Explain my nitrogen result.",
                    context="Nitrogen=12 kg/ha (Low); this is user-entered.",
                )

        messages=post.call_args.kwargs["json"]["messages"]
        context_message=next(
            message for message in messages
            if message["role"]=="system" and "user-provided context" in message["content"]
        )
        self.assertIn("Do not invent missing measurements",context_message["content"])
        self.assertIn("Nitrogen=12 kg/ha",context_message["content"])

    def test_missing_key_keeps_known_faq_answers(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("How do I choose fertilizer for my crop?")

        self.assertIn("soil test",result)

    def test_crop_detection_question_gets_an_offline_explanation(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("What is crop detection?")

        self.assertIn("close-up photo of one leaf",result)
        self.assertIn("possible disease",result)
        self.assertIn("38 classes across 14 crops",result)

    def test_disease_count_question_gets_model_scope_offline(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("How many plant diseases are there?")

        self.assertIn("38 classes across 14 crops",result)
        self.assertIn("26 disease-labelled classes and 12 healthy classes",result)
        self.assertIn("not the total number of plant diseases",result)

    def test_crop_detection_faq_falls_back_when_groq_returns_404(self):
        response=Mock()
        response.status_code=404
        response.json.return_value={"error":{"message":"Requested model is unavailable"}}
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                result=chatbot.answer("What is crop detection?")

        self.assertIn("close-up photo of one leaf",result)
        self.assertNotIn("HTTP 404",result)

    def test_groq_404_detail_is_reported_for_unmatched_questions(self):
        response=Mock()
        response.status_code=404
        response.json.return_value={"error":{"message":"Requested model is unavailable"}}
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                with self.assertRaisesRegex(
                    chatbot.ChatbotError,
                    "HTTP 404.*Requested model is unavailable",
                ):
                    chatbot.answer("How can I manage intercropping in my field?")

    def test_crop_rotation_question_gets_an_offline_explanation(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("What is crop rotation and why is it important?")

        self.assertIn("growing different crops",result)
        self.assertIn("pest and disease cycles",result)
        self.assertIn("legume",result)

    def test_matching_local_knowledge_is_available_without_api_key(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("What agricultural services does IMD provide farmers?")

        self.assertIn("Gramin Krishi Mausam Seva",result)
        self.assertIn("Source: IMD Meteorological Services for Agriculture",result)
        self.assertIn("https://mausam.imd.gov.in/",result)

    def test_retrieved_sources_are_added_to_groq_prompt(self):
        response=Mock()
        response.raise_for_status.return_value=None
        response.json.return_value={
            "choices":[{"message":{"content":"IMD provides agromet advisories."}}]
        }

        with patch.dict(os.environ,{"GROQ_API_KEY":"test-key"}):
            with patch.object(chatbot.requests,"post",return_value=response) as post:
                chatbot.answer("What services does IMD provide for agricultural weather?")

        messages=post.call_args.kwargs["json"]["messages"]
        knowledge_message=next(
            message for message in messages
            if message["role"]=="system" and "retrieved passages" in message["content"]
        )
        self.assertIn("meteorological-agriculture-services.php",knowledge_message["content"])
        self.assertIn("cite their source title and URL",knowledge_message["content"])

    def test_wheat_season_question_gets_offline_answer(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("In which season can we crop wheat?")

        self.assertIn("Rabi crop",result)
        self.assertIn("October",result)

    def test_wheat_question_falls_back_when_groq_rejects_key(self):
        response=Mock()
        response.status_code=401
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"invalid-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                result=chatbot.answer("In which season can we crop wheat?")

        self.assertIn("Rabi crop",result)
        self.assertNotIn("Groq rejected",result)

    def test_missing_key_explains_how_to_enable_open_ended_answers(self):
        with patch.object(chatbot,"_get_api_key",return_value=""):
            result=chatbot.answer("How can I improve soil organic matter?")

        self.assertIn("GROQ_API_KEY",result)

    def test_invalid_api_key_is_reported(self):
        response=Mock()
        response.status_code=401
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"invalid-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                with self.assertRaisesRegex(chatbot.ChatbotError,"Groq rejected the API key"):
                    chatbot.answer("How should I prepare my field?")

    def test_rust_question_gets_local_guidance_when_api_key_is_invalid(self):
        response=Mock()
        response.status_code=401
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"invalid-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                result=chatbot.answer("What fertilizer can I use for rust disease?")

        self.assertIn("Fertilizer does not cure rust",result)
        self.assertNotIn("Note:",result)

    def test_invalid_key_error_points_to_secrets_and_environment(self):
        response=Mock()
        response.status_code=401
        response.raise_for_status.side_effect=requests.HTTPError(response=response)

        with patch.dict(os.environ,{"GROQ_API_KEY":"invalid-key"}):
            with patch.object(chatbot.requests,"post",return_value=response):
                with self.assertRaisesRegex(chatbot.ChatbotError,"\\.streamlit/secrets.toml"):
                    chatbot.answer("How can I control a severe locust outbreak?")


if __name__=="__main__":
    unittest.main()
