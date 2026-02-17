#!/usr/bin/env python3
"""
Demo script showing enhanced language metadata with sample data
Since we don't have the DiACL dataset, this creates sample data to demonstrate the enhancement
"""

import json
from datetime import datetime
from enhanced_metadata import LanguageMetadataEnhancer
from enhanced_anki import EnhancedLanguageAnkiGenerator

def create_sample_language_cache():
    """Create sample language cache to demonstrate enhancements"""
    
    sample_cache = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_languages': 5,
            'total_regions': 12
        },
        'languages': {
            'sample_001': {
                'presence_id': 'sample_001',
                'language_id': '1001',
                'name': 'Romansh',
                'iso639_3': 'roh',
                'glottocode': 'roma1326',
                'alternative_names': 'Rumantsch, Rhaeto-Romance, Grischun',
                'focal_point': {'lat': 46.6, 'lon': 9.8},
                'language_area': 'Europe',
                'reliability': 'High',
                'wkt_url': 'https://example.com/romansh.wkt',
                'original_wkt': 'MULTIPOLYGON(((9.5 46.5, 10.2 46.5, 10.2 46.8, 9.5 46.8, 9.5 46.5)))',
                'original_polygon_count': 15,
                'region_count': 3,
                'polygon_groups_wkt': [],
                'region_info': [
                    {
                        'region_index': 1,
                        'polygon_count': 8,
                        'total_area_km2': 2341.5,
                        'centroid_lat': 46.65,
                        'centroid_lon': 9.85,
                        'bounds': {'min_lon': 9.3, 'min_lat': 46.4, 'max_lon': 10.1, 'max_lat': 46.9}
                    },
                    {
                        'region_index': 2,
                        'polygon_count': 4,
                        'total_area_km2': 892.3,
                        'centroid_lat': 46.72,
                        'centroid_lon': 9.95,
                        'bounds': {'min_lon': 9.6, 'min_lat': 46.6, 'max_lon': 10.3, 'max_lat': 46.8}
                    }
                ],
                'time_frame': {'start': '800 CE', 'end': 'present'},
                'source_references': ['Swiss Federal Statistical Office', 'UNESCO Atlas']
            },
            'sample_002': {
                'presence_id': 'sample_002',
                'language_id': '1002',
                'name': 'Cornish',
                'iso639_3': 'cor',
                'glottocode': 'corn1251',
                'alternative_names': 'Kernewek, Cornic',
                'focal_point': {'lat': 50.2, 'lon': -5.3},
                'language_area': 'Europe',
                'reliability': 'High',
                'wkt_url': 'https://example.com/cornish.wkt',
                'original_wkt': 'MULTIPOLYGON(((-5.7 50.0, -4.9 50.0, -4.9 50.4, -5.7 50.4, -5.7 50.0)))',
                'original_polygon_count': 1,
                'region_count': 1,
                'polygon_groups_wkt': [],
                'region_info': [
                    {
                        'region_index': 1,
                        'polygon_count': 1,
                        'total_area_km2': 3563.2,
                        'centroid_lat': 50.2,
                        'centroid_lon': -5.3,
                        'bounds': {'min_lon': -5.7, 'min_lat': 50.0, 'max_lon': -4.9, 'max_lat': 50.4}
                    }
                ],
                'time_frame': {'start': '600 CE', 'end': '1777'},
                'source_references': ['Cornish Language Partnership', 'UNESCO Atlas']
            },
            'sample_003': {
                'presence_id': 'sample_003', 
                'language_id': '1003',
                'name': 'Mandarin Chinese',
                'iso639_3': 'cmn',
                'glottocode': 'mand1415',
                'alternative_names': '官话, Guanhua, Standard Chinese, Putonghua',
                'focal_point': {'lat': 39.9, 'lon': 116.4},
                'language_area': 'Asia',
                'reliability': 'High',
                'wkt_url': 'https://example.com/mandarin.wkt',
                'original_wkt': 'MULTIPOLYGON(((73 18, 135 18, 135 54, 73 54, 73 18)))',
                'original_polygon_count': 45,
                'region_count': 8,
                'polygon_groups_wkt': [],
                'region_info': [
                    {
                        'region_index': 1,
                        'polygon_count': 25,
                        'total_area_km2': 8500000.0,
                        'centroid_lat': 35.5,
                        'centroid_lon': 104.2,
                        'bounds': {'min_lon': 73.0, 'min_lat': 18.0, 'max_lon': 135.0, 'max_lat': 54.0}
                    }
                ],
                'time_frame': {'start': '1000 BCE', 'end': 'present'},
                'source_references': ['Ministry of Education China', 'Ethnologue']
            },
            'sample_004': {
                'presence_id': 'sample_004',
                'language_id': '1004', 
                'name': 'Ainu',
                'iso639_3': 'ain',
                'glottocode': 'ainu1240',
                'alternative_names': 'アイヌ語, Aynu',
                'focal_point': {'lat': 43.2, 'lon': 142.8},
                'language_area': 'Asia',
                'reliability': 'Medium',
                'wkt_url': 'https://example.com/ainu.wkt',
                'original_wkt': 'MULTIPOLYGON(((140 42, 146 42, 146 46, 140 46, 140 42)))',
                'original_polygon_count': 8,
                'region_count': 2,
                'polygon_groups_wkt': [],
                'region_info': [
                    {
                        'region_index': 1,
                        'polygon_count': 6,
                        'total_area_km2': 78421.2,
                        'centroid_lat': 43.2,
                        'centroid_lon': 142.8,
                        'bounds': {'min_lon': 140.0, 'min_lat': 42.0, 'max_lon': 146.0, 'max_lat': 46.0}
                    }
                ],
                'time_frame': {'start': '1000 CE', 'end': 'present'},
                'source_references': ['Foundation for Research and Promotion of Ainu Culture']
            },
            'sample_005': {
                'presence_id': 'sample_005',
                'language_id': '1005',
                'name': 'Basque',
                'iso639_3': 'eus', 
                'glottocode': 'basq1248',
                'alternative_names': 'Euskera, Euskara',
                'focal_point': {'lat': 43.0, 'lon': -2.0},
                'language_area': 'Europe',
                'reliability': 'High',
                'wkt_url': 'https://example.com/basque.wkt',
                'original_wkt': 'MULTIPOLYGON(((-3.0 42.8, -1.5 42.8, -1.5 43.5, -3.0 43.5, -3.0 42.8)))',
                'original_polygon_count': 12,
                'region_count': 3,
                'polygon_groups_wkt': [],
                'region_info': [
                    {
                        'region_index': 1,
                        'polygon_count': 8,
                        'total_area_km2': 20664.0,
                        'centroid_lat': 43.0,
                        'centroid_lon': -2.25,
                        'bounds': {'min_lon': -3.0, 'min_lat': 42.8, 'max_lon': -1.5, 'max_lat': 43.5}
                    }
                ],
                'time_frame': {'start': '200 BCE', 'end': 'present'},
                'source_references': ['Euskaltzaindia', 'Government of Navarre']
            }
        },
        'processing_errors': []
    }
    
    return sample_cache

def add_sample_enhanced_metadata(cache_data):
    """Add sample enhanced metadata to demonstrate capabilities"""
    
    # Sample enhanced metadata for each language
    enhanced_metadata_samples = {
        'sample_001': {  # Romansh
            'speakers': {
                'total_speakers': 60000,
                'native_speakers': 60000,
                'l2_speakers': 10000,
                'speaker_estimate_year': 2020,
                'speaker_trend': 'declining',
                'vitality_status': 'definitely endangered'
            },
            'periods': {
                'first_attested': '800 CE',
                'active_period_start': '800 CE', 
                'active_period_end': None,
                'extinction_date': None,
                'revival_attempts': ['Swiss Federal promotion programs'],
                'historical_stages': ['Old Romansh (800-1500)', 'Modern Romansh (1500-present)']
            },
            'classification': {
                'language_family': 'Indo-European',
                'subfamily': 'Italic',
                'branch': 'Romance',
                'group': 'Rhaeto-Romance',
                'macro_family': None,
                'isolate': False
            },
            'geographic_details': {
                'countries': ['Switzerland'],
                'regions': ['Graubünden'],
                'major_cities': ['Chur', 'Davos'],
                'geographic_spread': 'regional',
                'migration_patterns': []
            },
            'linguistic_features': {
                'writing_systems': ['Latin script'],
                'official_status': ['Switzerland (federal level)', 'Graubünden (canton level)'],
                'standardized': True,
                'dialects': ['Sursilvan', 'Sutsilvan', 'Surmiran', 'Puter', 'Vallader'],
                'mutual_intelligibility': {'Italian': 'partial'},
                'typological_features': {}
            },
            'cultural_context': {
                'cultural_significance': 'Swiss cultural identity, Alpine heritage',
                'literature_tradition': True,
                'oral_tradition': True, 
                'media_presence': ['Radio Rumantsch', 'Televisiun Rumantscha'],
                'education_status': 'Primary education available'
            }
        },
        'sample_002': {  # Cornish
            'speakers': {
                'total_speakers': 2000,
                'native_speakers': 0,
                'l2_speakers': 2000,
                'speaker_estimate_year': 2021,
                'speaker_trend': 'increasing',
                'vitality_status': 'extinct'
            },
            'periods': {
                'first_attested': '600 CE',
                'active_period_start': '600 CE',
                'active_period_end': '1777',
                'extinction_date': '1777',
                'revival_attempts': ['Cornish Revival (20th century)', 'Cornish Language Partnership'],
                'historical_stages': ['Old Cornish (600-1100)', 'Middle Cornish (1100-1600)', 'Late Cornish (1600-1777)']
            },
            'classification': {
                'language_family': 'Indo-European',
                'subfamily': 'Celtic',
                'branch': 'Brythonic',
                'group': 'Southwestern Brythonic',
                'macro_family': None,
                'isolate': False
            },
            'geographic_details': {
                'countries': ['United Kingdom'],
                'regions': ['Cornwall'],
                'major_cities': ['Truro', 'St Austell'],
                'geographic_spread': 'local',
                'migration_patterns': ['Medieval spread throughout Cornwall']
            },
            'linguistic_features': {
                'writing_systems': ['Latin script'],
                'official_status': ['Cornwall (regional recognition)'],
                'standardized': 'partial',
                'dialects': ['Eastern Cornish', 'Western Cornish'],
                'mutual_intelligibility': {'Welsh': 'partial', 'Breton': 'partial'},
                'typological_features': {}
            },
            'cultural_context': {
                'cultural_significance': 'Celtic heritage, Cornish identity revival',
                'literature_tradition': True,
                'oral_tradition': True,
                'media_presence': ['Online resources', 'Cultural events'],
                'education_status': 'Revival classes, some schools'
            }
        },
        'sample_003': {  # Mandarin Chinese
            'speakers': {
                'total_speakers': 918000000,
                'native_speakers': 918000000,
                'l2_speakers': 200000000,
                'speaker_estimate_year': 2020,
                'speaker_trend': 'stable',
                'vitality_status': 'safe'
            },
            'periods': {
                'first_attested': '1000 BCE',
                'active_period_start': '1000 BCE',
                'active_period_end': None,
                'extinction_date': None,
                'revival_attempts': [],
                'historical_stages': ['Classical Chinese (1000 BCE-220 CE)', 'Modern Chinese (1919-present)']
            },
            'classification': {
                'language_family': 'Sino-Tibetan',
                'subfamily': 'Chinese',
                'branch': 'Sinitic',
                'group': 'Mandarin',
                'macro_family': None,
                'isolate': False
            },
            'geographic_details': {
                'countries': ['China', 'Taiwan', 'Singapore'],
                'regions': ['Northern China', 'Northeastern China', 'Southwestern China'],
                'major_cities': ['Beijing', 'Shanghai', 'Tianjin', 'Chengdu'],
                'geographic_spread': 'international',
                'migration_patterns': ['Historical southward expansion', 'Modern global diaspora']
            },
            'linguistic_features': {
                'writing_systems': ['Simplified Chinese', 'Traditional Chinese'],
                'official_status': ['China', 'Taiwan', 'Singapore'],
                'standardized': True,
                'dialects': ['Beijing dialect', 'Northeastern Mandarin', 'Southwestern Mandarin'],
                'mutual_intelligibility': {'Cantonese': 'limited'},
                'typological_features': {'tone_count': 4}
            },
            'cultural_context': {
                'cultural_significance': 'Chinese civilization, official language',
                'literature_tradition': True,
                'oral_tradition': True,
                'media_presence': ['CCTV', 'China Radio International', 'Extensive internet'],
                'education_status': 'Primary education language'
            }
        },
        'sample_004': {  # Ainu
            'speakers': {
                'total_speakers': 10,
                'native_speakers': 2,
                'l2_speakers': 8,
                'speaker_estimate_year': 2021,
                'speaker_trend': 'critically endangered',
                'vitality_status': 'critically endangered'
            },
            'periods': {
                'first_attested': '1000 CE',
                'active_period_start': '1000 CE',
                'active_period_end': None,
                'extinction_date': None,
                'revival_attempts': ['Ainu language revitalization programs', 'University courses'],
                'historical_stages': ['Classical Ainu (1000-1800)', 'Modern Ainu (1800-present)']
            },
            'classification': {
                'language_family': None,
                'subfamily': None,
                'branch': None,
                'group': None,
                'macro_family': None,
                'isolate': True
            },
            'geographic_details': {
                'countries': ['Japan', 'Russia'],
                'regions': ['Hokkaido', 'Sakhalin'],
                'major_cities': ['Sapporo'],
                'geographic_spread': 'regional',
                'migration_patterns': ['Historically throughout northern Japan']
            },
            'linguistic_features': {
                'writing_systems': ['Katakana', 'Latin script'],
                'official_status': ['Japan (indigenous language recognition)'],
                'standardized': 'partial',
                'dialects': ['Hokkaido Ainu', 'Sakhalin Ainu'],
                'mutual_intelligibility': {},
                'typological_features': {}
            },
            'cultural_context': {
                'cultural_significance': 'Indigenous Ainu culture, bear ceremonies',
                'literature_tradition': False,
                'oral_tradition': True,
                'media_presence': ['Cultural preservation programs'],
                'education_status': 'University courses, cultural programs'
            }
        },
        'sample_005': {  # Basque
            'speakers': {
                'total_speakers': 750000,
                'native_speakers': 665000,
                'l2_speakers': 85000,
                'speaker_estimate_year': 2019,
                'speaker_trend': 'stable',
                'vitality_status': 'vulnerable'
            },
            'periods': {
                'first_attested': '200 BCE',
                'active_period_start': '200 BCE',
                'active_period_end': None,
                'extinction_date': None,
                'revival_attempts': ['Basque language revival (1960s-present)'],
                'historical_stages': ['Proto-Basque (pre-200 BCE)', 'Classical Basque (200 BCE-1500 CE)', 'Modern Basque (1500-present)']
            },
            'classification': {
                'language_family': None,
                'subfamily': None,
                'branch': None,
                'group': None,
                'macro_family': None,
                'isolate': True
            },
            'geographic_details': {
                'countries': ['Spain', 'France'],
                'regions': ['Basque Country', 'Navarre', 'Pyrénées-Atlantiques'],
                'major_cities': ['Bilbao', 'San Sebastián', 'Vitoria-Gasteiz', 'Pamplona'],
                'geographic_spread': 'regional',
                'migration_patterns': ['Historical spread across Pyrenees', 'Modern diaspora']
            },
            'linguistic_features': {
                'writing_systems': ['Latin script'],
                'official_status': ['Basque Country (Spain)', 'Navarre (co-official)', 'Pyrénées-Atlantiques (France)'],
                'standardized': True,
                'dialects': ['Bizkaian', 'Gipuzkoan', 'Upper Navarrese', 'Lapurdian', 'Lower Navarrese', 'Zuberoan'],
                'mutual_intelligibility': {},
                'typological_features': {'ergative_case': True}
            },
            'cultural_context': {
                'cultural_significance': 'Basque cultural identity, pre-Indo-European heritage',
                'literature_tradition': True,
                'oral_tradition': True,
                'media_presence': ['ETB (Basque television)', 'Radio Euskadi', 'Euskaldunon Egunkaria'],
                'education_status': 'Full education in Basque available'
            }
        }
    }
    
    # Add enhanced metadata to each language
    for lang_id, lang_data in cache_data['languages'].items():
        if lang_id in enhanced_metadata_samples:
            lang_data['enhanced_metadata'] = enhanced_metadata_samples[lang_id]
        else:
            # Add empty structure for languages without sample data
            enhancer = LanguageMetadataEnhancer()
            lang_data['enhanced_metadata'] = enhancer.get_enhanced_metadata_structure()
    
    return cache_data

def main():
    """Run the demo"""
    print("🎭 Enhanced Language Metadata Demo")
    print("=" * 50)
    
    # Create sample data
    print("📝 Creating sample language cache...")
    cache_data = create_sample_language_cache()
    
    # Add enhanced metadata
    print("🔧 Adding enhanced metadata...")
    enhanced_cache = add_sample_enhanced_metadata(cache_data)
    
    # Save sample cache
    cache_file = "demo_enhanced_cache.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cache, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Sample enhanced cache saved to {cache_file}")
    
    # Generate sample flashcards
    print("🎴 Generating demo flashcards...")
    try:
        generator = EnhancedLanguageAnkiGenerator(cache_file)
        deck_file = generator.generate_deck(output_file="demo_enhanced_languages.apkg")
        print(f"✅ Demo flashcards generated: {deck_file}")
    except Exception as e:
        print(f"❌ Error generating flashcards: {e}")
    
    # Show sample of enhanced data
    print("\n📊 Sample Enhanced Language Data:")
    print("-" * 40)
    
    sample_lang = enhanced_cache['languages']['sample_001']  # Romansh
    print(f"Language: {sample_lang['name']}")
    print(f"Family: {sample_lang['enhanced_metadata']['classification']['language_family']}")
    print(f"Speakers: {sample_lang['enhanced_metadata']['speakers']['total_speakers']:,}")
    print(f"Status: {sample_lang['enhanced_metadata']['speakers']['vitality_status']}")
    print(f"Trend: {sample_lang['enhanced_metadata']['speakers']['speaker_trend']}")
    print(f"Countries: {', '.join(sample_lang['enhanced_metadata']['geographic_details']['countries'])}")
    
    print("\n🎉 Demo complete! Check the generated files:")
    print(f"  - {cache_file} (sample enhanced language data)")  
    print(f"  - demo_enhanced_languages.apkg (sample Anki deck)")
    print("\nTo see the full enhancement structure, open the JSON file!")

if __name__ == "__main__":
    main()