#!/usr/bin/env python3
"""
Comprehensive Language Learning Tool
Demonstrates working with the enhanced geographical + linguistic dataset
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

class LanguageLearningTool:
    def __init__(self):
        self.load_enhanced_cache()
        
    def load_enhanced_cache(self):
        """Load the enhanced language cache"""
        cache_file = Path('enhanced_language_cache.json')
        if not cache_file.exists():
            print("❌ Enhanced cache not found. Please run enhance_language_cache.py first.")
            return
            
        with open(cache_file, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)
        self.languages = self.cache['languages']
        print(f"✅ Loaded {len(self.languages)} languages")
    
    def get_languages_by_criteria(self, 
                                 family: Optional[str] = None,
                                 difficulty: Optional[str] = None,
                                 vitality: Optional[str] = None,
                                 min_speakers: Optional[int] = None,
                                 max_speakers: Optional[int] = None,
                                 writing_system: Optional[str] = None) -> List[Dict]:
        """Filter languages by various criteria"""
        results = []
        
        for lang_id, lang in self.languages.items():
            # Check family
            if family and (not lang.get('family') or lang['family']['name'].lower() != family.lower()):
                continue
                
            # Check difficulty
            if difficulty and (lang.get('difficulty_level') or '').lower() != difficulty.lower():
                continue
                
            # Check vitality
            if vitality and (lang.get('vitality') or '').lower() != vitality.lower():
                continue
                
            # Check speaker count
            speakers = lang.get('speakers')
            if min_speakers or max_speakers:
                if not speakers:  # Skip if no speaker data when filtering by speakers
                    continue
                if min_speakers and speakers < min_speakers:
                    continue
                if max_speakers and speakers > max_speakers:
                    continue
                    
            # Check writing system
            if writing_system and (lang.get('writing_system') or '').lower() != writing_system.lower():
                continue
            
            lang['_id'] = lang_id
            results.append(lang)
            
        return results
    
    def generate_anki_cards(self, languages: List[Dict], card_type: str = "basic") -> List[Dict]:
        """Generate different types of Anki cards"""
        cards = []
        
        for lang in languages:
            if card_type == "basic":
                cards.extend(self._generate_basic_cards(lang))
            elif card_type == "geographic":
                cards.extend(self._generate_geographic_cards(lang))
            elif card_type == "linguistic":
                cards.extend(self._generate_linguistic_cards(lang))
            elif card_type == "cultural":
                cards.extend(self._generate_cultural_cards(lang))
                
        return cards
    
    def _generate_basic_cards(self, lang: Dict) -> List[Dict]:
        """Basic language identification cards"""
        cards = []
        
        # Language name -> ISO code
        cards.append({
            'front': f"What is the ISO 639-3 code for {lang['name']}?",
            'back': lang['iso639_3'],
            'tags': ['iso-codes', 'basic', lang['iso639_3'].lower()]
        })
        
        # ISO code -> Language name  
        cards.append({
            'front': f"What language does '{lang['iso639_3']}' represent?",
            'back': lang['name'],
            'tags': ['iso-codes', 'basic', lang['iso639_3'].lower()]
        })
        
        return cards
    
    def _generate_geographic_cards(self, lang: Dict) -> List[Dict]:
        """Geography-focused cards"""
        cards = []
        geo = lang.get('geographic_summary', {})
        
        if geo:
            # Area question
            if geo.get('total_area_km2'):
                cards.append({
                    'front': f"What is the approximate total area where {lang['name']} is spoken?",
                    'back': f"{geo['total_area_km2']:,.1f} km²",
                    'tags': ['geography', 'area', lang['iso639_3'].lower()]
                })
            
            # Location question
            if geo.get('avg_latitude') and geo.get('avg_longitude'):
                lat = geo['avg_latitude']
                lon = geo['avg_longitude']
                hemisphere_ns = "Northern" if lat > 0 else "Southern"
                hemisphere_ew = "Eastern" if lon > 0 else "Western"
                
                cards.append({
                    'front': f"In which hemispheres is {lang['name']} primarily spoken?",
                    'back': f"{hemisphere_ns} and {hemisphere_ew} hemispheres<br>({lat:.2f}°, {lon:.2f}°)",
                    'tags': ['geography', 'location', lang['iso639_3'].lower()]
                })
        
        # Countries
        if lang.get('countries'):
            cards.append({
                'front': f"In which countries is {lang['name']} official or widely spoken?",
                'back': ', '.join(lang['countries']),
                'tags': ['geography', 'countries', lang['iso639_3'].lower()]
            })
            
        return cards
    
    def _generate_linguistic_cards(self, lang: Dict) -> List[Dict]:
        """Linguistics-focused cards"""
        cards = []
        
        # Family information
        if lang.get('family'):
            family = lang['family']
            cards.append({
                'front': f"What language family does {lang['name']} belong to?",
                'back': f"{family['name']} ({family['level']})" + 
                       (f" of {family['parent']}" if family.get('parent') else ""),
                'tags': ['linguistics', 'family', lang['iso639_3'].lower()]
            })
        
        # Writing system
        if lang.get('writing_system'):
            cards.append({
                'front': f"What writing system does {lang['name']} use?",
                'back': lang['writing_system'],
                'tags': ['linguistics', 'writing', lang['iso639_3'].lower()]
            })
        
        # Word order
        if lang.get('word_order'):
            cards.append({
                'front': f"What is the basic word order of {lang['name']}?",
                'back': f"{lang['word_order']} (Subject-Verb-Object type)",
                'tags': ['linguistics', 'syntax', lang['iso639_3'].lower()]
            })
        
        # Morphology
        if lang.get('morphology'):
            cards.append({
                'front': f"What type of morphology does {lang['name']} have?",
                'back': lang['morphology'].capitalize(),
                'tags': ['linguistics', 'morphology', lang['iso639_3'].lower()]
            })
            
        return cards
    
    def _generate_cultural_cards(self, lang: Dict) -> List[Dict]:
        """Culture and context cards"""
        cards = []
        
        # Speaker count
        if lang.get('speakers'):
            cards.append({
                'front': f"Approximately how many speakers does {lang['name']} have?",
                'back': f"{lang['speakers']:,} speakers",
                'tags': ['culture', 'speakers', lang['iso639_3'].lower()]
            })
        
        # Vitality status
        if lang.get('vitality') and lang['vitality'] != 'unknown':
            cards.append({
                'front': f"What is the vitality status of {lang['name']}?",
                'back': lang['vitality'].capitalize(),
                'tags': ['culture', 'vitality', lang['iso639_3'].lower()]
            })
        
        # Cultural notes
        if lang.get('cultural_notes'):
            cards.append({
                'front': f"What are some key cultural aspects of {lang['name']}?",
                'back': lang['cultural_notes'],
                'tags': ['culture', 'notes', lang['iso639_3'].lower()]
            })
            
        return cards
    
    def show_language_summary(self, lang: Dict):
        """Show a comprehensive summary of a language"""
        print(f"\n🌍 {lang['name']} ({lang['iso639_3']})")
        print("=" * 50)
        
        # Basic info
        if lang.get('alternative_names'):
            print(f"📝 Alternative names: {', '.join(lang['alternative_names'])}")
        
        # Family
        if lang.get('family'):
            family = lang['family']
            family_str = family['name']
            if family.get('parent'):
                family_str += f" ({family['parent']})"
            print(f"🌳 Language family: {family_str}")
        
        # Speakers & vitality
        if lang.get('speakers'):
            print(f"👥 Speakers: {lang['speakers']:,}")
        if lang.get('vitality') and lang['vitality'] != 'unknown':
            vitality_emoji = {'safe': '🟢', 'vulnerable': '🟡', 'endangered': '🟠', 'extinct': '🔴'}.get(lang['vitality'], '⚪')
            print(f"{vitality_emoji} Vitality: {lang['vitality']}")
        
        # Linguistic features
        if lang.get('writing_system'):
            print(f"✍️ Writing system: {lang['writing_system']}")
        if lang.get('word_order'):
            print(f"📝 Word order: {lang['word_order']}")
        if lang.get('morphology'):
            print(f"🔤 Morphology: {lang['morphology']}")
        
        # Geography
        if lang.get('countries'):
            print(f"🗺️ Countries: {', '.join(lang['countries'])}")
        
        geo = lang.get('geographic_summary', {})
        if geo:
            if geo.get('total_area_km2'):
                print(f"📏 Total area: {geo['total_area_km2']:,.1f} km²")
            if geo.get('avg_latitude') and geo.get('avg_longitude'):
                print(f"📍 Center point: {geo['avg_latitude']:.2f}°, {geo['avg_longitude']:.2f}°")
        
        # Cultural notes
        if lang.get('cultural_notes'):
            print(f"💡 Cultural notes: {lang['cultural_notes']}")
        
        print()
    
    def demo_basic_queries(self):
        """Demonstrate basic query capabilities"""
        print("🔍 Basic Query Demonstrations\n")
        
        # Major world languages
        major_langs = self.get_languages_by_criteria(min_speakers=100000000)
        print(f"🌐 Major world languages (100M+ speakers): {len(major_langs)}")
        for lang in sorted(major_langs, key=lambda x: x.get('speakers') or 0, reverse=True)[:5]:
            print(f"  • {lang['name']}: {lang.get('speakers', 'unknown'):,} speakers")
        
        # Germanic languages
        germanic_langs = self.get_languages_by_criteria(family="Germanic")
        print(f"\n🇩🇪 Germanic languages: {len(germanic_langs)}")
        for lang in germanic_langs[:5]:
            print(f"  • {lang['name']} ({lang['iso639_3']})")
        
        # Endangered languages
        endangered_langs = self.get_languages_by_criteria(vitality="endangered")
        print(f"\n⚠️ Endangered languages: {len(endangered_langs)}")
        for lang in endangered_langs[:3]:
            print(f"  • {lang['name']}: {lang.get('speakers', 'unknown')} speakers")
        
        # Languages with Latin script
        latin_langs = self.get_languages_by_criteria(writing_system="Latin")
        print(f"\n📝 Languages using Latin script: {len(latin_langs)}")
        
    def demo_anki_generation(self):
        """Demonstrate Anki card generation"""
        print("\n🎴 Anki Card Generation Demo\n")
        
        # Get a few interesting languages
        demo_langs = []
        
        # Add English for comprehensiveness
        eng = [l for l in self.languages.values() if l['iso639_3'] == 'ENG']
        if eng:
            eng[0]['_id'] = 'eng'
            demo_langs.append(eng[0])
        
        # Add Latvian for regional interest
        lav = [l for l in self.languages.values() if l['iso639_3'] == 'LAV']
        if lav:
            lav[0]['_id'] = 'lav'
            demo_langs.append(lav[0])
        
        # Add Basque for uniqueness
        baq = [l for l in self.languages.values() if l['iso639_3'] == 'BAQ']
        if baq:
            baq[0]['_id'] = 'baq'
            demo_langs.append(baq[0])
        
        for card_type in ['basic', 'geographic', 'linguistic', 'cultural']:
            print(f"📋 {card_type.title()} Cards:")
            cards = self.generate_anki_cards(demo_langs, card_type)
            
            for card in cards[:3]:  # Show first 3 of each type
                print(f"  Q: {card['front']}")
                print(f"  A: {card['back']}")
                print(f"  Tags: {', '.join(card['tags'])}\n")
    
    def interactive_explorer(self):
        """Simple interactive exploration"""
        print("\n🔍 Interactive Language Explorer")
        print("Commands: family <name>, difficulty <level>, vitality <status>, speakers <min>-<max>, random, quit")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'random':
                    lang = random.choice(list(self.languages.values()))
                    self.show_language_summary(lang)
                elif cmd.startswith('family '):
                    family = cmd[7:].strip()
                    langs = self.get_languages_by_criteria(family=family)
                    print(f"Found {len(langs)} {family} languages:")
                    for lang in langs[:5]:
                        print(f"  • {lang['name']} - {lang.get('speakers', 'unknown')} speakers")
                elif cmd.startswith('difficulty '):
                    difficulty = cmd[11:].strip()
                    langs = self.get_languages_by_criteria(difficulty=difficulty)
                    print(f"Found {len(langs)} {difficulty} languages:")
                    for lang in langs[:5]:
                        print(f"  • {lang['name']}")
                elif cmd.startswith('vitality '):
                    vitality = cmd[9:].strip()
                    langs = self.get_languages_by_criteria(vitality=vitality)
                    print(f"Found {len(langs)} {vitality} languages:")
                    for lang in langs[:5]:
                        print(f"  • {lang['name']}")
                else:
                    print("Unknown command. Try: family germanic, difficulty easy, vitality safe, random, quit")
                    
            except KeyboardInterrupt:
                break
        
        print("👋 Goodbye!")

def main():
    """Main demonstration function"""
    print("🌍 Comprehensive Language Learning Tool")
    print("=" * 50)
    
    tool = LanguageLearningTool()
    
    # Run demonstrations
    tool.demo_basic_queries()
    tool.demo_anki_generation()
    
    # Show detailed examples
    print("\n📖 Detailed Language Examples:")
    
    # Show English
    eng = [l for l in tool.languages.values() if l['iso639_3'] == 'ENG']
    if eng:
        tool.show_language_summary(eng[0])
    
    # Show Latvian (regional)
    lav = [l for l in tool.languages.values() if l['iso639_3'] == 'LAV']
    if lav:
        tool.show_language_summary(lav[0])
    
    # Show Basque (unique)
    baq = [l for l in tool.languages.values() if l['iso639_3'] == 'BAQ']
    if baq:
        tool.show_language_summary(baq[0])
    
    # Optional interactive mode
    try:
        explore = input("\n❓ Would you like to explore interactively? (y/N): ").strip().lower()
        if explore in ['y', 'yes']:
            tool.interactive_explorer()
    except KeyboardInterrupt:
        pass
    
    print("\n✅ Demonstration complete!")

if __name__ == "__main__":
    main()