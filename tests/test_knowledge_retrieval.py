import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import knowledge_retrieval


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_weather_advisory_question_retrieves_the_cited_imd_source(self):
        chunks=knowledge_retrieval.retrieve_knowledge(
            "How does IMD's weather advisory help farmers?"
        )

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["title"]=="IMD Meteorological Services for Agriculture" for chunk in chunks))
        self.assertTrue(all(chunk["url"].startswith("https://mausam.imd.gov.in/") for chunk in chunks))

    def test_unrelated_question_does_not_receive_imd_context(self):
        chunks=knowledge_retrieval.retrieve_knowledge(
            "How do I identify rust disease in wheat?"
        )

        self.assertEqual(chunks,[])

    def test_unsourced_chunk_raises_a_clear_knowledge_base_error(self):
        with tempfile.TemporaryDirectory() as directory:
            chunks_path=Path(directory)/"chunks.json"
            manifest_path=Path(directory)/"source_manifest.json"
            chunks_path.write_text(
                '[{"title":"Test","url":"https://example.com","text":"weather"}]',
                encoding="utf-8",
            )
            manifest_path.write_text("[]",encoding="utf-8")
            with patch.object(knowledge_retrieval,"CHUNKS_PATH",chunks_path):
                with patch.object(knowledge_retrieval,"MANIFEST_PATH",manifest_path):
                    with self.assertRaisesRegex(
                        knowledge_retrieval.KnowledgeBaseError,
                        "invalid or unsourced chunk",
                    ):
                        knowledge_retrieval.load_knowledge_chunks()


if __name__=="__main__":
    unittest.main()
