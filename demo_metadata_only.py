#!/usr/bin/env python3
"""
Demo script showing enhanced language metadata (without Anki generation)
Since genanki isn't installed, this just demonstrates the metadata enhancement
"""

import json
from datetime import datetime
from enhanced_metadata import LanguageMetadataEnhancer

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
                'time_frame': {'start': '600 CE', 'end': '1777'},
                'region_info': [
                    {
                        'region_index': 1,
                        'total_area_km2': 3563.2,
                        'centroid_lat': 50.2,
                        'centroid_lon': -5.3
                    }
                ]
            }
        },
        'processing_errors': []
    }
    
    return sample_cache

def add_sample_enhanced_metadata(cache_data):
    """Add sample enhanced metadata to demonstrate capabilities"""
    
    # Sample enhanced metadata
    enhanced_samples = {
        'sample_001': {  # Romansh
            'speakers': {
                'total_speakers': 60000,
                'native_speakers': 60000,
                'speaker_trend': 'declining',
                'vitality_status': 'definitely endangered'
            },
            'classification': {
                'language_family': 'Indo-European',
                'subfamily': 'Romance',
                'isolate': False
            },
            'geographic_details': {
                'countries': ['Switzerland'],
                'regions': ['Graubünden'],
                'geographic_spread': 'regional'
            }
        },
        'sample_002': {  # Cornish
            'speakers': {
                'total_speakers': 2000,
                'native_speakers': 0,
                'speaker_trend': 'increasing',
                'vitality_status': 'extinct'
            },
            'classification': {
                'language_family': 'Indo-European',
                'subfamily': 'Celtic',
                'isolate': False
            },
            'geographic_details': {
                'countries': ['United Kingdom'],
                'regions': ['Cornwall'],
                'geographic_spread': 'local'
            }
        }
    }
    
    # Add enhanced metadata to each language
    enhancer = LanguageMetadataEnhancer()
    for lang_id, lang_data in cache_data['languages'].items():
        # Start with empty structure
        lang_data['enhanced_metadata'] = enhancer.get_enhanced_metadata_structure()
        
        # Add sample data if available
        if lang_id in enhanced_samples:
            sample_data = enhanced_samples[lang_id]
            for category, category_data in sample_data.items():
                if category in lang_data['enhanced_metadata']:
                    lang_data['enhanced_metadata'][category].update(category_data)
    
    return cache_data

def main():
    """Run the metadata demo"""
    print("🎭 Enhanced Language Metadata Demo")
    print("=" * 50)
    
    # Create sample data
    print("📝 Creating sample language cache...")
    cache_data = create_sample_language_cache()
    
    # Show original structure
    print(f"Original languages: {len(cache_data['languages'])}")
    
    # Add enhanced metadata
    print("🔧 Adding enhanced metadata...")
    enhanced_cache = add_sample_enhanced_metadata(cache_data)
    
    # Save sample cache
    cache_file = "demo_metadata_sample.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cache, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Sample enhanced cache saved to {cache_file}")
    
    # Show sample of enhanced data structure
    print("\n📊 Enhanced Metadata Structure:")
    print("-" * 40)
    
    enhancer = LanguageMetadataEnhancer()
    structure = enhancer.get_enhanced_metadata_structure()
    
    print("Main categories added:")
    for category in structure:
        subcategories = list(structure[category].keys())
        print(f"  🏷️  {category}: {len(subcategories)} fields")
        print(f"      └── {', '.join(subcategories[:3])}{'...' if len(subcategories) > 3 else ''}")
    
    print("\n📝 Sample Language Data (Romansh):")
    print("-" * 40)
    
    romansh = enhanced_cache['languages']['sample_001']
    metadata = romansh['enhanced_metadata']
    
    print(f"Language: {romansh['name']} ({romansh['iso639_3']})")
    print(f"Family: {metadata['classification']['language_family']} > {metadata['classification']['subfamily']}")
    print(f"Speakers: {metadata['speakers']['total_speakers']:,} ({metadata['speakers']['vitality_status']})")
    print(f"Countries: {', '.join(metadata['geographic_details']['countries'])}")
    print(f"Trend: {metadata['speakers']['speaker_trend']}")
    print(f"Language Isolate: {metadata['classification']['isolate']}")
    
    print("\n📝 Sample Language Data (Cornish):")
    print("-" * 40)
    
    cornish = enhanced_cache['languages']['sample_002']
    metadata = cornish['enhanced_metadata']
    
    print(f"Language: {cornish['name']} ({cornish['iso639_3']})")
    print(f"Family: {metadata['classification']['language_family']} > {metadata['classification']['subfamily']}")
    print(f"Speakers: {metadata['speakers']['total_speakers']:,} ({metadata['speakers']['vitality_status']})")
    print(f"Countries: {', '.join(metadata['geographic_details']['countries'])}")
    print(f"Trend: {metadata['speakers']['speaker_trend']} (revival language)")
    
    print("\n🎯 What This Enhancement Adds:")
    print("-" * 40)
    print("✅ Speaker demographics (total, native, L2, trends)")
    print("✅ Vitality status (UNESCO endangerment scale)")  
    print("✅ Language classification (family, subfamily, isolates)")
    print("✅ Geographic details (countries, regions, spread)")
    print("✅ Temporal information (periods, attestation, extinction)")
    print("✅ Linguistic features (scripts, official status, dialects)")
    print("✅ Cultural context (literature, education, media)")
    print("✅ Comprehensive metadata sources framework")
    
    print("\n🚀 Next Steps:")
    print("-" * 40)
    print("1. Install genanki: pip install genanki")
    print("2. Get DiACL dataset (countries.json)")
    print("3. Run: python enhanced_metadata.py")
    print("4. Run: python enhanced_anki.py")
    print("5. Import generated .apkg into Anki!")
    
    print(f"\n📄 Generated files:")
    print(f"  - {cache_file} (sample metadata structure)")
    print(f"  - Check ENHANCEMENT_REPORT.md for full details")

if __name__ == "__main__":
    main()