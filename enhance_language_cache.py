#!/usr/bin/env python3
"""
Enhanced Language Cache Creator
Combines rich geographical language data with comprehensive linguistic metadata
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class LanguageFamily:
    name: str
    level: str  # "family", "subfamily", "branch"
    parent: Optional[str] = None

@dataclass
class EnhancedLanguageMetadata:
    # Core identification
    name: str
    iso639_3: str
    glottocode: Optional[str] = None
    alternative_names: List[str] = None
    
    # Geographic info (from original dataset)
    regions: List[Dict] = None
    region_polygons: List[List] = None
    time_frame: Dict = None
    
    # Enhanced metadata
    family: Optional[LanguageFamily] = None
    speakers: Optional[int] = None
    vitality: str = "unknown"  # endangered, vulnerable, safe, extinct, etc.
    writing_system: Optional[str] = None
    
    # Linguistic features
    word_order: Optional[str] = None  # SOV, SVO, VSO, etc.
    morphology: Optional[str] = None  # agglutinative, fusional, isolating, etc.
    difficulty_level: str = "intermediate"
    
    # Cultural context
    countries: List[str] = None
    cultural_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.alternative_names is None:
            self.alternative_names = []
        if self.regions is None:
            self.regions = []
        if self.region_polygons is None:
            self.region_polygons = []
        if self.countries is None:
            self.countries = []

class LanguageEnhancer:
    def __init__(self):
        self.language_families = self._init_language_families()
        self.language_metadata = self._init_language_metadata()
        
    def _init_language_families(self) -> Dict[str, LanguageFamily]:
        """Initialize major language families"""
        families = {
            # Indo-European
            'eng': LanguageFamily('Germanic', 'subfamily', 'Indo-European'),
            'deu': LanguageFamily('Germanic', 'subfamily', 'Indo-European'),
            'swe': LanguageFamily('Germanic', 'subfamily', 'Indo-European'),
            'rus': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'ukr': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'pol': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'ces': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'bul': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'hrv': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'srp': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'slv': LanguageFamily('Slavic', 'subfamily', 'Indo-European'),
            'fra': LanguageFamily('Romance', 'subfamily', 'Indo-European'),
            'spa': LanguageFamily('Romance', 'subfamily', 'Indo-European'),
            'ita': LanguageFamily('Romance', 'subfamily', 'Indo-European'),
            'por': LanguageFamily('Romance', 'subfamily', 'Indo-European'),
            'rom': LanguageFamily('Romance', 'subfamily', 'Indo-European'),
            'lat': LanguageFamily('Italic', 'subfamily', 'Indo-European'),
            'gre': LanguageFamily('Hellenic', 'subfamily', 'Indo-European'),
            'hye': LanguageFamily('Armenian', 'subfamily', 'Indo-European'),
            'lav': LanguageFamily('Baltic', 'subfamily', 'Indo-European'),
            'lit': LanguageFamily('Baltic', 'subfamily', 'Indo-European'),
            
            # Sino-Tibetan
            'cmn': LanguageFamily('Chinese', 'subfamily', 'Sino-Tibetan'),
            'yue': LanguageFamily('Chinese', 'subfamily', 'Sino-Tibetan'),
            'bod': LanguageFamily('Tibetic', 'subfamily', 'Sino-Tibetan'),
            
            # Afroasiatic
            'ara': LanguageFamily('Semitic', 'subfamily', 'Afroasiatic'),
            'heb': LanguageFamily('Semitic', 'subfamily', 'Afroasiatic'),
            'amh': LanguageFamily('Semitic', 'subfamily', 'Afroasiatic'),
            
            # Niger-Congo
            'swa': LanguageFamily('Bantu', 'subfamily', 'Niger-Congo'),
            'yor': LanguageFamily('Volta-Niger', 'subfamily', 'Niger-Congo'),
            'ibo': LanguageFamily('Volta-Niger', 'subfamily', 'Niger-Congo'),
            
            # Language isolates and smaller families
            'baq': LanguageFamily('Basque', 'isolate', None),
            'kor': LanguageFamily('Koreanic', 'family', None),
            'jpn': LanguageFamily('Japonic', 'family', None),
            'fin': LanguageFamily('Uralic', 'family', None),
            'hun': LanguageFamily('Uralic', 'family', None),
        }
        return families
    
    def _init_language_metadata(self) -> Dict[str, Dict]:
        """Initialize enhanced metadata for major languages"""
        return {
            'eng': {
                'speakers': 1500000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'analytic',
                'difficulty_level': 'easy',
                'countries': ['US', 'UK', 'CA', 'AU', 'NZ', 'IE', 'ZA'],
                'cultural_notes': 'Global lingua franca, widely used in science, technology, and international communication'
            },
            'cmn': {
                'speakers': 918000000,
                'vitality': 'safe',
                'writing_system': 'Chinese characters',
                'word_order': 'SVO',
                'morphology': 'isolating',
                'difficulty_level': 'very_hard',
                'countries': ['CN', 'TW', 'SG'],
                'cultural_notes': 'Most spoken language in the world, tonal language with rich literary tradition'
            },
            'spa': {
                'speakers': 500000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'easy',
                'countries': ['ES', 'MX', 'AR', 'CO', 'PE', 'VE', 'CL', 'EC', 'GT', 'CU'],
                'cultural_notes': 'Second most spoken language by native speakers, official in 21 countries'
            },
            'rus': {
                'speakers': 258000000,
                'vitality': 'safe',
                'writing_system': 'Cyrillic',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'hard',
                'countries': ['RU', 'BY', 'KZ', 'KG'],
                'cultural_notes': 'Official language of Russia, widely used across former Soviet Union'
            },
            'fra': {
                'speakers': 280000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['FR', 'CA', 'BE', 'CH', 'LU', 'MC'],
                'cultural_notes': 'Language of diplomacy, widely used in international organizations'
            },
            'deu': {
                'speakers': 100000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'V2',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['DE', 'AT', 'CH', 'LI', 'LU', 'BE'],
                'cultural_notes': 'Major language of science and philosophy, complex grammar with cases'
            },
            'jpn': {
                'speakers': 125000000,
                'vitality': 'safe',
                'writing_system': 'Hiragana/Katakana/Kanji',
                'word_order': 'SOV',
                'morphology': 'agglutinative',
                'difficulty_level': 'very_hard',
                'countries': ['JP'],
                'cultural_notes': 'Complex honorific system, three writing systems, rich cultural heritage'
            },
            'por': {
                'speakers': 260000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['BR', 'PT', 'AO', 'MZ', 'GW', 'CV', 'ST', 'TL'],
                'cultural_notes': 'Official language of 9 countries, major language of South America and Africa'
            },
            'kor': {
                'speakers': 77000000,
                'vitality': 'safe',
                'writing_system': 'Hangul',
                'word_order': 'SOV',
                'morphology': 'agglutinative',
                'difficulty_level': 'hard',
                'countries': ['KR', 'KP'],
                'cultural_notes': 'Scientific alphabet system, complex honorific levels'
            },
            'ita': {
                'speakers': 65000000,
                'vitality': 'safe',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['IT', 'CH', 'SM', 'VA'],
                'cultural_notes': 'Language of art, music, and cuisine, closest to Latin'
            },
            'lav': {
                'speakers': 1750000,
                'vitality': 'vulnerable',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['LV'],
                'cultural_notes': 'One of two surviving Baltic languages, rich folklore tradition'
            },
            'lit': {
                'speakers': 3000000,
                'vitality': 'vulnerable',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'medium',
                'countries': ['LT'],
                'cultural_notes': 'Most conservative Indo-European language, seven-case system'
            },
            'baq': {
                'speakers': 750000,
                'vitality': 'vulnerable',
                'writing_system': 'Latin',
                'word_order': 'SOV',
                'morphology': 'agglutinative',
                'difficulty_level': 'very_hard',
                'countries': ['ES', 'FR'],
                'cultural_notes': 'Language isolate, pre-Indo-European, ergative-absolutive alignment'
            },
            'roh': {
                'speakers': 60000,
                'vitality': 'endangered',
                'writing_system': 'Latin',
                'word_order': 'SVO',
                'morphology': 'fusional',
                'difficulty_level': 'hard',
                'countries': ['CH'],
                'cultural_notes': 'Rhaeto-Romance language, official language of Switzerland'
            }
        }
    
    def enhance_language(self, lang_data: Dict) -> EnhancedLanguageMetadata:
        """Enhance a single language entry with additional metadata"""
        iso_code = lang_data.get('iso639_3', '').lower()
        
        # Extract region information
        regions = lang_data.get('region_info', [])
        polygons = lang_data.get('region_polygons', [])
        
        # Get enhanced metadata if available
        enhanced = self.language_metadata.get(iso_code, {})
        
        # Create enhanced language object
        enhanced_lang = EnhancedLanguageMetadata(
            name=lang_data.get('name', ''),
            iso639_3=iso_code.upper(),
            glottocode=lang_data.get('glottocode'),
            alternative_names=lang_data.get('alternative_names', '').split(', ') if lang_data.get('alternative_names') else [],
            regions=regions,
            region_polygons=polygons,
            time_frame=lang_data.get('time_frame', {}),
            family=self.language_families.get(iso_code),
            speakers=enhanced.get('speakers'),
            vitality=enhanced.get('vitality', 'unknown'),
            writing_system=enhanced.get('writing_system'),
            word_order=enhanced.get('word_order'),
            morphology=enhanced.get('morphology'),
            difficulty_level=enhanced.get('difficulty_level', 'intermediate'),
            countries=enhanced.get('countries', []),
            cultural_notes=enhanced.get('cultural_notes')
        )
        
        return enhanced_lang
    
    def calculate_geographic_summary(self, regions: List[Dict]) -> Dict:
        """Calculate geographic summary from region data"""
        if not regions:
            return {}
        
        total_area = sum(r.get('total_area_km2', 0) for r in regions)
        latitudes = [r.get('centroid_lat') for r in regions if r.get('centroid_lat')]
        longitudes = [r.get('centroid_lon') for r in regions if r.get('centroid_lon')]
        
        summary = {
            'total_area_km2': total_area,
            'region_count': len(regions)
        }
        
        if latitudes and longitudes:
            summary.update({
                'avg_latitude': sum(latitudes) / len(latitudes),
                'avg_longitude': sum(longitudes) / len(longitudes),
                'lat_range': [min(latitudes), max(latitudes)],
                'lon_range': [min(longitudes), max(longitudes)]
            })
        
        return summary

def main():
    """Main function to enhance the language cache"""
    print("🌍 Enhancing Language Cache with Metadata...")
    
    # Load original cache
    cache_file = Path('language_cache.json')
    if not cache_file.exists():
        print("❌ language_cache.json not found!")
        return
    
    print("📂 Loading language cache...")
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    
    enhancer = LanguageEnhancer()
    enhanced_languages = {}
    
    print(f"🔍 Processing {len(cache['languages'])} languages...")
    
    for lang_id, lang_data in cache['languages'].items():
        try:
            enhanced = enhancer.enhance_language(lang_data)
            
            # Convert to dict and add geographic summary
            lang_dict = asdict(enhanced)
            if enhanced.regions:
                lang_dict['geographic_summary'] = enhancer.calculate_geographic_summary(enhanced.regions)
            
            enhanced_languages[lang_id] = lang_dict
            
        except Exception as e:
            print(f"⚠️ Error processing language {lang_id}: {e}")
    
    # Create enhanced cache structure
    enhanced_cache = {
        'metadata': {
            'generated_at': cache['metadata']['generated_at'],
            'enhanced_at': '2026-02-17T15:30:00Z',
            'total_languages': len(enhanced_languages),
            'enhancement_version': '1.0'
        },
        'languages': enhanced_languages,
        'processing_info': {
            'enhanced_languages': len([l for l in enhanced_languages.values() if l.get('speakers')]),
            'languages_with_families': len([l for l in enhanced_languages.values() if l.get('family')]),
            'languages_with_regions': len([l for l in enhanced_languages.values() if l.get('regions')])
        }
    }
    
    # Save enhanced cache
    output_file = 'enhanced_language_cache.json'
    print(f"💾 Saving enhanced cache to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cache, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n✅ Enhancement Complete!")
    print(f"📊 Total languages: {len(enhanced_languages)}")
    print(f"🔤 With speaker counts: {enhanced_cache['processing_info']['enhanced_languages']}")
    print(f"🌳 With language families: {enhanced_cache['processing_info']['languages_with_families']}")
    print(f"🗺️ With geographic data: {enhanced_cache['processing_info']['languages_with_regions']}")
    
    # Show some examples
    print("\n🎯 Sample Enhanced Languages:")
    for lang_id, lang in list(enhanced_languages.items())[:5]:
        name = lang['name']
        family = lang.get('family', {}).get('name', 'Unknown') if lang.get('family') else 'Unknown'
        speakers = lang.get('speakers')
        regions = len(lang.get('regions', []))
        
        print(f"  • {name} ({lang['iso639_3']}): {family} family, {speakers or 'unknown'} speakers, {regions} regions")

if __name__ == "__main__":
    main()