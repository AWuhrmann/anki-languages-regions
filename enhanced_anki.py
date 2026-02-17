#!/usr/bin/env python3
"""
Enhanced Anki Flashcard Generator with Rich Language Metadata
Generates flashcards with comprehensive language information including speakers, periods, classification, etc.
"""

import genanki
import json
import os
from pathlib import Path
from datetime import datetime

class EnhancedLanguageAnkiGenerator:
    """Generate Anki flashcards with rich language metadata"""
    
    def __init__(self, enhanced_cache_file="enhanced_language_cache.json"):
        self.cache_file = enhanced_cache_file
        self.images_dir = Path("language_maps")
        self.deck_id = 3247432489  # Fixed deck ID
        self.model_id = 7970701730  # Enhanced model ID (different from basic)
        
        # Create enhanced card model
        self.model = self._create_enhanced_model()
        
    def _create_enhanced_model(self):
        """Create an enhanced Anki card model with metadata fields"""
        
        fields = [
            {'name': 'LanguageName'},
            {'name': 'Image'},
            {'name': 'ISO639'},
            {'name': 'Glottocode'},
            {'name': 'LanguageFamily'},
            {'name': 'TotalSpeakers'},
            {'name': 'NativeSpeakers'},
            {'name': 'SpeakerTrend'},
            {'name': 'VitalityStatus'},
            {'name': 'FirstAttested'},
            {'name': 'ActivePeriod'},
            {'name': 'Countries'},
            {'name': 'GeographicSpread'},
            {'name': 'WritingSystems'},
            {'name': 'OfficialStatus'},
            {'name': 'CulturalSignificance'},
            {'name': 'AlternativeNames'},
            {'name': 'RegionInfo'},
        ]
        
        templates = [
            {
                'name': 'Map to Language (Basic)',
                'qfmt': '''
                    <div class="map-container">
                        {{Image}}
                    </div>
                    <div class="question-text">
                        What language is primarily spoken in these regions?
                    </div>
                ''',
                'afmt': '''
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="language-name">{{LanguageName}}</div>
                    <div class="basic-info">
                        {{#ISO639}}<span class="tag">ISO: {{ISO639}}</span>{{/ISO639}}
                        {{#LanguageFamily}}<span class="tag family">{{LanguageFamily}}</span>{{/LanguageFamily}}
                    </div>
                    {{#TotalSpeakers}}<div class="speakers">👥 {{TotalSpeakers}} speakers</div>{{/TotalSpeakers}}
                    {{#Countries}}<div class="countries">🌍 {{Countries}}</div>{{/Countries}}
                ''',
            },
            {
                'name': 'Language to Details (Advanced)',
                'qfmt': '''
                    <div class="language-header">
                        <h2>{{LanguageName}}</h2>
                        {{#ISO639}}<span class="iso">{{ISO639}}</span>{{/ISO639}}
                    </div>
                    <div class="question-text">
                        What do you know about this language?
                    </div>
                ''',
                'afmt': '''
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="detailed-info">
                        {{Image}}
                        
                        <div class="info-section">
                            <h3>🗣️ Speakers</h3>
                            {{#TotalSpeakers}}<p><strong>Total:</strong> {{TotalSpeakers}}</p>{{/TotalSpeakers}}
                            {{#NativeSpeakers}}<p><strong>Native:</strong> {{NativeSpeakers}}</p>{{/NativeSpeakers}}
                            {{#SpeakerTrend}}<p><strong>Trend:</strong> {{SpeakerTrend}}</p>{{/SpeakerTrend}}
                            {{#VitalityStatus}}<p><strong>Status:</strong> {{VitalityStatus}}</p>{{/VitalityStatus}}
                        </div>
                        
                        <div class="info-section">
                            <h3>🏛️ Classification</h3>
                            {{#LanguageFamily}}<p><strong>Family:</strong> {{LanguageFamily}}</p>{{/LanguageFamily}}
                        </div>
                        
                        <div class="info-section">
                            <h3>📅 Historical Period</h3>
                            {{#FirstAttested}}<p><strong>First attested:</strong> {{FirstAttested}}</p>{{/FirstAttested}}
                            {{#ActivePeriod}}<p><strong>Active period:</strong> {{ActivePeriod}}</p>{{/ActivePeriod}}
                        </div>
                        
                        <div class="info-section">
                            <h3>🌍 Geography</h3>
                            {{#Countries}}<p><strong>Countries:</strong> {{Countries}}</p>{{/Countries}}
                            {{#GeographicSpread}}<p><strong>Spread:</strong> {{GeographicSpread}}</p>{{/GeographicSpread}}
                        </div>
                        
                        {{#WritingSystems}}
                        <div class="info-section">
                            <h3>✍️ Writing & Status</h3>
                            <p><strong>Scripts:</strong> {{WritingSystems}}</p>
                            {{#OfficialStatus}}<p><strong>Official in:</strong> {{OfficialStatus}}</p>{{/OfficialStatus}}
                        </div>
                        {{/WritingSystems}}
                        
                        {{#CulturalSignificance}}
                        <div class="info-section">
                            <h3>🎭 Cultural Context</h3>
                            <p>{{CulturalSignificance}}</p>
                        </div>
                        {{/CulturalSignificance}}
                        
                        {{#AlternativeNames}}
                        <div class="info-section">
                            <h3>📛 Alternative Names</h3>
                            <p>{{AlternativeNames}}</p>
                        </div>
                        {{/AlternativeNames}}
                    </div>
                ''',
            },
            {
                'name': 'Speakers & Vitality',
                'qfmt': '''
                    <div class="language-header">
                        <h2>{{LanguageName}}</h2>
                        {{Image}}
                    </div>
                    <div class="question-text">
                        How many people speak this language? What's its vitality status?
                    </div>
                ''',
                'afmt': '''
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="vitality-info">
                        {{#TotalSpeakers}}
                        <div class="speaker-count">
                            <h3>👥 {{TotalSpeakers}} total speakers</h3>
                            {{#NativeSpeakers}}<p>Native speakers: {{NativeSpeakers}}</p>{{/NativeSpeakers}}
                        </div>
                        {{/TotalSpeakers}}
                        
                        {{#VitalityStatus}}
                        <div class="vitality-status">
                            <h3>📊 Vitality Status</h3>
                            <p class="status-{{VitalityStatus}}">{{VitalityStatus}}</p>
                        </div>
                        {{/VitalityStatus}}
                        
                        {{#SpeakerTrend}}
                        <div class="trend">
                            <h3>📈 Trend</h3>
                            <p class="trend-{{SpeakerTrend}}">{{SpeakerTrend}}</p>
                        </div>
                        {{/SpeakerTrend}}
                        
                        {{#Countries}}<p><strong>Spoken in:</strong> {{Countries}}</p>{{/Countries}}
                    </div>
                ''',
            },
        ]
        
        css = """
        .card { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            font-size: 16px;
            line-height: 1.4;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .map-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        img { 
            max-width: 100%; 
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .question-text {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin: 20px 0;
        }
        
        .language-name {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin: 20px 0;
        }
        
        .language-header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .language-header h2 {
            margin: 0;
            color: #2c3e50;
        }
        
        .iso {
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
        
        .basic-info {
            text-align: center;
            margin: 15px 0;
        }
        
        .tag {
            display: inline-block;
            background: #ecf0f1;
            color: #2c3e50;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 2px 4px;
            font-size: 12px;
        }
        
        .tag.family {
            background: #e74c3c;
            color: white;
        }
        
        .speakers {
            text-align: center;
            font-size: 18px;
            color: #27ae60;
            margin: 10px 0;
        }
        
        .countries {
            text-align: center;
            color: #7f8c8d;
            margin: 10px 0;
        }
        
        .detailed-info {
            text-align: left;
        }
        
        .info-section {
            background: #f8f9fa;
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .info-section h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 16px;
        }
        
        .info-section p {
            margin: 5px 0;
        }
        
        .vitality-info {
            text-align: center;
        }
        
        .speaker-count h3 {
            color: #27ae60;
            margin-bottom: 10px;
        }
        
        .status-extinct { color: #e74c3c; font-weight: bold; }
        .status-critically-endangered { color: #e67e22; font-weight: bold; }
        .status-severely-endangered { color: #f39c12; font-weight: bold; }
        .status-definitely-endangered { color: #f1c40f; font-weight: bold; }
        .status-vulnerable { color: #27ae60; font-weight: bold; }
        .status-safe { color: #2ecc71; font-weight: bold; }
        
        .trend-declining { color: #e74c3c; }
        .trend-stable { color: #f39c12; }
        .trend-increasing { color: #27ae60; }
        
        @media (max-width: 600px) {
            .card { padding: 10px; }
            .language-name { font-size: 24px; }
        }
        """
        
        return genanki.Model(
            self.model_id,
            'Enhanced Language Model',
            fields=fields,
            templates=templates,
            css=css
        )
    
    def load_enhanced_cache(self):
        """Load enhanced language cache"""
        if not Path(self.cache_file).exists():
            raise FileNotFoundError(f"Enhanced cache file {self.cache_file} not found. Run enhanced_metadata.py first.")
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_speaker_count(self, count):
        """Format speaker count for display"""
        if not count or count == 0:
            return None
        
        try:
            num = int(count)
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num/1000:.0f}K"
            else:
                return f"{num:,}"
        except:
            return str(count)
    
    def format_countries(self, countries_list):
        """Format countries list for display"""
        if not countries_list:
            return None
        
        if isinstance(countries_list, list):
            if len(countries_list) <= 3:
                return ", ".join(countries_list)
            else:
                return f"{', '.join(countries_list[:3])} +{len(countries_list)-3} more"
        
        return str(countries_list)
    
    def format_period(self, start, end):
        """Format time period for display"""
        if not start and not end:
            return None
        
        if start and end:
            return f"{start} - {end}"
        elif start:
            return f"{start} - present"
        elif end:
            return f"until {end}"
        
        return None
    
    def create_note_from_language(self, lang_id, lang_data):
        """Create an Anki note from enhanced language data"""
        
        # Basic information
        name = lang_data.get('name', 'Unknown Language')
        iso639 = lang_data.get('iso639_3', '')
        glottocode = lang_data.get('glottocode', '')
        
        # Enhanced metadata
        metadata = lang_data.get('enhanced_metadata', {})
        speakers = metadata.get('speakers', {})
        periods = metadata.get('periods', {})
        classification = metadata.get('classification', {})
        geographic = metadata.get('geographic_details', {})
        linguistic = metadata.get('linguistic_features', {})
        cultural = metadata.get('cultural_context', {})
        
        # Format fields for Anki
        fields = {
            'LanguageName': name,
            'Image': '',  # Will be filled by image processing
            'ISO639': iso639,
            'Glottocode': glottocode,
            'LanguageFamily': classification.get('language_family', ''),
            'TotalSpeakers': self.format_speaker_count(speakers.get('total_speakers')),
            'NativeSpeakers': self.format_speaker_count(speakers.get('native_speakers')),
            'SpeakerTrend': speakers.get('speaker_trend', ''),
            'VitalityStatus': speakers.get('vitality_status', ''),
            'FirstAttested': periods.get('first_attested', ''),
            'ActivePeriod': self.format_period(
                periods.get('active_period_start'), 
                periods.get('active_period_end')
            ),
            'Countries': self.format_countries(geographic.get('countries')),
            'GeographicSpread': geographic.get('geographic_spread', ''),
            'WritingSystems': ', '.join(linguistic.get('writing_systems', [])) if linguistic.get('writing_systems') else '',
            'OfficialStatus': ', '.join(linguistic.get('official_status', [])) if linguistic.get('official_status') else '',
            'CulturalSignificance': cultural.get('cultural_significance', ''),
            'AlternativeNames': lang_data.get('alternative_names', ''),
            'RegionInfo': self._format_region_info(lang_data.get('region_info', [])),
        }
        
        # Clean up empty fields
        cleaned_fields = []
        for field_def in self.model.fields:
            field_name = field_def['name']
            value = fields.get(field_name, '')
            if value is None:
                value = ''
            cleaned_fields.append(str(value))
        
        return genanki.Note(
            model=self.model,
            fields=cleaned_fields,
            tags=[f"language:{lang_id}", "enhanced-metadata"]
        )
    
    def _format_region_info(self, region_info):
        """Format region info for display"""
        if not region_info:
            return ''
        
        total_area = sum(region.get('total_area_km2', 0) for region in region_info)
        region_count = len(region_info)
        
        return f"{region_count} regions, {total_area:.0f} km² total"
    
    def generate_deck(self, max_languages=None, output_file="enhanced_languages.apkg"):
        """Generate enhanced Anki deck"""
        
        print("🎴 Generating enhanced language flashcards...")
        
        # Load enhanced cache
        cache_data = self.load_enhanced_cache()
        
        # Create deck
        deck = genanki.Deck(
            deck_id=self.deck_id,
            name=f'Enhanced Language Regions ({datetime.now().strftime("%Y-%m-%d")})'
        )
        
        languages = list(cache_data['languages'].items())
        if max_languages:
            languages = languages[:max_languages]
        
        print(f"Creating flashcards for {len(languages)} languages...")
        
        notes_created = 0
        for i, (lang_id, lang_data) in enumerate(languages):
            try:
                note = self.create_note_from_language(lang_id, lang_data)
                deck.add_note(note)
                notes_created += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Created {i + 1}/{len(languages)} notes...")
                    
            except Exception as e:
                print(f"❌ Error creating note for {lang_data.get('name', lang_id)}: {e}")
        
        # Save deck
        genanki.Package(deck).write_to_file(output_file)
        
        print(f"✅ Enhanced deck saved to {output_file}")
        print(f"   Languages: {len(languages)}")
        print(f"   Notes created: {notes_created}")
        print(f"   Card types: {len(self.model.templates)}")
        
        return output_file

def main():
    """Main function to generate enhanced language deck"""
    
    generator = EnhancedLanguageAnkiGenerator()
    
    try:
        deck_file = generator.generate_deck(max_languages=100)  # Limit for testing
        print(f"\n🎉 Enhanced language deck ready: {deck_file}")
        print("\nCard types included:")
        print("1. Map to Language (Basic) - Shows map, asks for language name")
        print("2. Language to Details (Advanced) - Shows language name, reveals comprehensive info")
        print("3. Speakers & Vitality - Focus on demographic and vitality information")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nTo generate enhanced flashcards:")
        print("1. First run the basic language cache generation")
        print("2. Then run: python enhanced_metadata.py")
        print("3. Finally run: python enhanced_anki.py")

if __name__ == "__main__":
    main()