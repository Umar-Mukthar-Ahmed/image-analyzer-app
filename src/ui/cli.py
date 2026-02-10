"""
Command-Line Interface for Image Analyzer App
"""
import sys
import logging
from pathlib import Path
from typing import Optional
from src.core.analyzer import ImageAnalyzer, AnalysisError
from src.decision.confidence_engine import ConfidenceEngine
from config.settings import settings

logger = logging.getLogger(__name__)


class ImageAnalyzerCLI:
    """Command-line interface for image analysis"""

    def __init__(self):
        """Initialize CLI"""
        self.analyzer = ImageAnalyzer()
        self.confidence_engine = ConfidenceEngine()
        self.running = True

    def display_banner(self):
        """Display application banner"""
        print("\n" + "="*60)
        print("  📸  IMAGE ANALYZER APP  📸")
        print("="*60)
        print("  Powered by Azure Computer Vision AI")
        print("="*60 + "\n")

    def display_menu(self):
        """Display main menu"""
        print("\n📋 Main Menu:")
        print("-" * 60)
        print("  1. 🔍 Analyze Image (Full Analysis)")
        print("  2. 📝 Extract Text (OCR)")
        print("  3. 📦 Detect Objects")
        print("  4. 💬 Generate Description")
        print("  5. 🏷️  Get Tags")
        print("  6. ❌ Exit")
        print("-" * 60)

    def get_choice(self) -> Optional[int]:
        """Get user menu choice"""
        try:
            choice = input("\nSelect option (1-6): ").strip()
            if choice in ['exit', 'quit', 'q']:
                return 6
            return int(choice)
        except ValueError:
            print("❌ Invalid input. Please enter a number 1-6.")
            return None

    def get_image_path(self) -> Optional[str]:
        """Get image path from user"""
        print("\n" + "-"*60)
        path = input("Enter image path (or 'back' to return): ").strip()

        if path.lower() in ['back', 'b', 'exit']:
            return None

        # Remove quotes if present
        path = path.strip('"').strip("'")

        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            return None

        return path

    def display_full_analysis(self, result):
        """Display complete analysis results"""
        print("\n" + "="*60)
        print("  ✅ ANALYSIS COMPLETE")
        print("="*60 + "\n")

        # Get summary
        summary = self.analyzer.get_analysis_summary(result)

        # Reliability Score
        reliability = summary['reliability']
        print(f"📊 Overall Reliability: {self.confidence_engine.format_confidence_display(reliability['reliability_score'])}")
        print(f"   Features Detected: {reliability['total_features']}")
        print(f"   High Confidence: {reliability['high_confidence_features']}")
        print(f"   Low Confidence: {reliability['low_confidence_features']}")

        # Descriptions
        if result.has_descriptions:
            print("\n💬 Description:")
            print("-" * 60)
            for i, desc in enumerate(result.descriptions[:3], 1):
                conf_display = self.confidence_engine.format_confidence_display(desc.confidence)
                print(f"  {i}. {desc.text}")
                print(f"     Confidence: {conf_display}")

        # Tags
        if result.tags:
            print("\n🏷️  Tags:")
            print("-" * 60)
            for tag in result.tags[:10]:
                conf_display = self.confidence_engine.format_confidence_display(tag.confidence)
                print(f"  • {tag.name:<20} {conf_display}")

        # Objects
        if result.has_objects:
            print("\n📦 Objects Detected:")
            print("-" * 60)
            for obj in result.objects:
                conf_display = self.confidence_engine.format_confidence_display(obj.confidence)
                print(f"  • {obj.name:<20} {conf_display}")

        # OCR Text
        if result.has_text:
            print("\n📝 Extracted Text:")
            print("-" * 60)
            for i, ocr in enumerate(result.ocr_text[:10], 1):
                conf_display = self.confidence_engine.format_confidence_display(ocr.confidence)
                print(f"  {i}. {ocr.text}")
                print(f"     Confidence: {conf_display}")

            if len(result.ocr_text) > 10:
                print(f"  ... and {len(result.ocr_text) - 10} more lines")

        # Faces
        if result.has_faces:
            print("\n👤 Faces Detected:")
            print("-" * 60)
            for i, face in enumerate(result.faces, 1):
                age_info = f"Age: ~{face.age}" if face.age else "Age: Unknown"
                gender_info = f"Gender: {face.gender}" if face.gender else ""
                print(f"  {i}. {age_info} {gender_info}")

        # Colors
        if result.dominant_colors:
            print("\n🎨 Color Information:")
            print("-" * 60)
            print(f"  Dominant Colors: {', '.join(result.dominant_colors)}")
            if result.accent_color:
                print(f"  Accent Color: {result.accent_color}")
            if result.is_bw_image:
                print(f"  Black & White: Yes")

        # Content Moderation
        if result.adult_score > 0.3 or result.racy_score > 0.3:
            print("\n⚠️  Content Moderation:")
            print("-" * 60)
            if result.is_adult_content:
                print(f"  Adult Content: Yes (Score: {result.adult_score:.2f})")
            if result.is_racy_content:
                print(f"  Racy Content: Yes (Score: {result.racy_score:.2f})")

        # Uncertainty Warnings
        warnings = summary['warnings']
        if warnings:
            print("\n⚠️  Uncertainty Notes:")
            print("-" * 60)
            for warning in warnings:
                print(f"  {warning}")

        # Suggestions
        suggestions = summary['suggestions']
        if suggestions:
            print("\n💡 Suggestions for Better Results:")
            print("-" * 60)
            for suggestion in suggestions:
                print(f"  {suggestion}")

        print("\n" + "="*60 + "\n")

    def display_ocr_results(self, ocr_data):
        """Display OCR results"""
        print("\n" + "="*60)
        print("  ✅ TEXT EXTRACTION COMPLETE")
        print("="*60 + "\n")

        decision = ocr_data['decision']

        print(f"📊 Confidence: {self.confidence_engine.format_confidence_display(ocr_data['confidence'])}")
        print(f"📝 Total Lines: {ocr_data['total_lines']}")
        print(f"\n💭 Assessment: {decision.recommendation}")

        if decision.uncertainty_note:
            print(f"⚠️  Note: {decision.uncertainty_note}")

        print("\n📄 Extracted Text:")
        print("-" * 60)

        if ocr_data['text']:
            print(ocr_data['text'])
        else:
            print("  No text detected in image.")

        print("\n" + "="*60 + "\n")

    def display_objects(self, objects_data):
        """Display object detection results"""
        print("\n" + "="*60)
        print("  ✅ OBJECT DETECTION COMPLETE")
        print("="*60 + "\n")

        print(f"📦 Total Objects: {objects_data['total_objects']}")

        if objects_data['objects']:
            print("\n🔍 Detected Objects:")
            print("-" * 60)

            for item in objects_data['objects']:
                obj = item['object']
                decision = item['decision']

                conf_display = self.confidence_engine.format_confidence_display(obj.confidence)
                print(f"\n  • {obj.name}")
                print(f"    Confidence: {conf_display}")
                print(f"    Assessment: {decision.recommendation}")

                if decision.uncertainty_note:
                    print(f"    ⚠️  {decision.uncertainty_note}")
        else:
            print("\n  No objects detected in image.")

        print("\n" + "="*60 + "\n")

    def display_description(self, desc_data):
        """Display description results"""
        print("\n" + "="*60)
        print("  ✅ DESCRIPTION GENERATED")
        print("="*60 + "\n")

        if desc_data['primary_description']:
            desc = desc_data['primary_description']
            decision = desc_data['decision']

            conf_display = self.confidence_engine.format_confidence_display(desc.confidence)

            print(f"💬 Description:")
            print(f"   \"{desc.text}\"")
            print(f"\n📊 Confidence: {conf_display}")
            print(f"💭 Assessment: {decision.recommendation}")

            if decision.uncertainty_note:
                print(f"⚠️  Note: {decision.uncertainty_note}")

            # Show alternative descriptions
            if len(desc_data['descriptions']) > 1:
                print("\n🔄 Alternative Descriptions:")
                print("-" * 60)
                for i, alt_desc in enumerate(desc_data['descriptions'][1:3], 2):
                    conf_display = self.confidence_engine.format_confidence_display(alt_desc.confidence)
                    print(f"  {i}. {alt_desc.text}")
                    print(f"     {conf_display}")
        else:
            print("  No description could be generated.")

        print("\n" + "="*60 + "\n")

    def run_full_analysis(self):
        """Run full image analysis"""
        image_path = self.get_image_path()
        if not image_path:
            return

        print("\n⏳ Analyzing image... Please wait...")

        try:
            result = self.analyzer.analyze_image(image_path, include_ocr=True)
            self.display_full_analysis(result)

            # Ask to save results
            self.offer_save_results(result, image_path)

        except AnalysisError as e:
            print(f"\n❌ Analysis Error: {str(e)}\n")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}\n")
            logger.exception("Unexpected error in full analysis")

    def run_ocr(self):
        """Run OCR text extraction"""
        image_path = self.get_image_path()
        if not image_path:
            return

        print("\n⏳ Extracting text... Please wait...")

        try:
            ocr_data = self.analyzer.extract_text_only(image_path)
            self.display_ocr_results(ocr_data)

        except AnalysisError as e:
            print(f"\n❌ OCR Error: {str(e)}\n")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}\n")

    def run_object_detection(self):
        """Run object detection"""
        image_path = self.get_image_path()
        if not image_path:
            return

        print("\n⏳ Detecting objects... Please wait...")

        try:
            objects_data = self.analyzer.detect_objects_only(image_path)
            self.display_objects(objects_data)

        except AnalysisError as e:
            print(f"\n❌ Detection Error: {str(e)}\n")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}\n")

    def run_description(self):
        """Run description generation"""
        image_path = self.get_image_path()
        if not image_path:
            return

        print("\n⏳ Generating description... Please wait...")

        try:
            desc_data = self.analyzer.get_description_only(image_path)
            self.display_description(desc_data)

        except AnalysisError as e:
            print(f"\n❌ Description Error: {str(e)}\n")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}\n")

    def run_tagging(self):
        """Run image tagging"""
        image_path = self.get_image_path()
        if not image_path:
            return

        print("\n⏳ Getting tags... Please wait...")

        try:
            result = self.analyzer.analyze_image(image_path, include_ocr=False)

            print("\n" + "="*60)
            print("  ✅ TAGGING COMPLETE")
            print("="*60 + "\n")

            if result.tags:
                print("🏷️  Tags:")
                print("-" * 60)
                for tag in result.tags:
                    conf_display = self.confidence_engine.format_confidence_display(tag.confidence)
                    print(f"  • {tag.name:<20} {conf_display}")
            else:
                print("  No tags generated.")

            print("\n" + "="*60 + "\n")

        except AnalysisError as e:
            print(f"\n❌ Tagging Error: {str(e)}\n")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {str(e)}\n")

    def offer_save_results(self, result, image_path):
        """Offer to save results to file"""
        save = input("Save results to file? (y/n): ").strip().lower()

        if save == 'y':
            filename = input("Enter filename (default: analysis_result.txt): ").strip()
            if not filename:
                filename = "analysis_result.txt"

            if not filename.endswith('.txt'):
                filename += '.txt'

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Image Analysis Results\n")
                    f.write(f"Image: {image_path}\n")
                    f.write("="*60 + "\n\n")

                    # Write JSON format
                    import json
                    f.write(json.dumps(result.to_dict(), indent=2))

                print(f"\n✅ Results saved to: {filename}\n")
            except Exception as e:
                print(f"\n❌ Error saving file: {str(e)}\n")

    def run(self):
        """Run interactive CLI"""
        self.display_banner()

        while self.running:
            self.display_menu()
            choice = self.get_choice()

            if choice is None:
                continue

            if choice == 1:
                self.run_full_analysis()
            elif choice == 2:
                self.run_ocr()
            elif choice == 3:
                self.run_object_detection()
            elif choice == 4:
                self.run_description()
            elif choice == 5:
                self.run_tagging()
            elif choice == 6:
                print("\n👋 Thank you for using Image Analyzer App!\n")
                self.running = False
            else:
                print("\n❌ Invalid choice. Please select 1-6.\n")


def main():
    """Main entry point for CLI"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.get_log_file_path()),
            logging.StreamHandler()
        ]
    )

    try:
        settings.validate()
    except ValueError as e:
        print("\n" + "="*60)
        print("  ❌ CONFIGURATION ERROR")
        print("="*60)
        print(f"\n{str(e)}\n")
        sys.exit(1)

    try:
        cli = ImageAnalyzerCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error in CLI")
        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()