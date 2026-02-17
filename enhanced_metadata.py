#!/usr/bin/env python3
"""
Enhanced Language Metadata Module
Extends the existing language data structure with additional metadata like speakers, periods, etc.
"""

import json
import requests
from datetime import datetime
from pathlib import Path

class LanguageMetadataEnhancer:
    """Enhanced metadata collector for language data"""
    
    def __init__(self):
        self.ethnologue_api_key = None  # Would need API key
        self.glottolog_cache = {}
        self.wals_cache = {}
        
    def get_enhanced_metadata_structure(self):
        """Return the enhanced metadata structure we want to add"""
        return {
            # Demographic data
            'speakers': {
                'total_speakers': None,           # Total number of speakers
                'native_speakers': None,          # Native speakers count
                'l2_speakers': None,              # Second language speakers
                'speaker_estimate_year': None,    # Year of estimate
                'speaker_trend': None,            # 'increasing', 'stable', 'declining', 'extinct'
                'vitality_status': None,          # UNESCO vitality scale
            },
            
            # Temporal data  
            'periods': {
                'first_attested': None,           # First written/recorded evidence
                'active_period_start': None,      # When actively spoken (start)
                'active_period_end': None,        # When actively spoken (end, if extinct)
                'extinction_date': None,          # If extinct, when
                'revival_attempts': [],           # Any revival efforts
                'historical_stages': [],         # Different historical periods
            },
            
            # Classification
            'classification': {
                'language_family': None,          # e.g. "Indo-European"
                'subfamily': None,               # e.g. "Germanic"
                'branch': None,                  # e.g. "West Germanic"  
                'group': None,                   # e.g. "High German"
                'macro_family': None,            # Larger groupings if proposed
                'isolate': False,                # True if language isolate
            },
            
            # Geographic details
            'geographic_details': {
                'countries': [],                 # List of countries where spoken
                'regions': [],                   # Specific regions/states
                'major_cities': [],              # Major cities where spoken
                'geographic_spread': None,       # 'local', 'regional', 'national', 'international'
                'migration_patterns': [],       # Historical movements
            },
            
            # Linguistic features
            'linguistic_features': {
                'writing_systems': [],           # Scripts used
                'official_status': [],           # Where it's official
                'standardized': None,            # True/False/Partial
                'dialects': [],                  # Major dialect groups
                'mutual_intelligibility': {},   # With related languages
                'typological_features': {},     # From WALS if available
            },
            
            # Cultural context
            'cultural_context': {
                'cultural_significance': None,   # Religious, ceremonial uses
                'literature_tradition': None,   # Written literature exists
                'oral_tradition': None,         # Oral literature/stories
                'media_presence': [],           # Radio, TV, internet
                'education_status': None,       # Used in schools
            },
            
            # Sources and reliability
            'metadata_sources': {
                'ethnologue_code': None,        # Ethnologue language code
                'iso639_1': None,              # ISO 639-1 (2-letter)
                'iso639_2': None,              # ISO 639-2 (3-letter)  
                'wals_code': None,             # World Atlas of Language Structures
                'multitree_code': None,        # MultiTree language code
                'sources': [],                 # List of data sources used
                'last_updated': None,          # When metadata was last updated
                'confidence_score': None,      # 0-1 confidence in the data
            }
        }
    
    def enhance_language_data(self, language_data):
        """Enhance existing language data with additional metadata"""
        enhanced = language_data.copy()
        
        # Add the enhanced metadata structure
        enhanced['enhanced_metadata'] = self.get_enhanced_metadata_structure()
        
        # Try to populate what we can from existing data
        self._populate_from_existing_data(enhanced)
        
        # Try to fetch additional data from external sources
        self._fetch_external_metadata(enhanced)
        
        return enhanced
    
    def _populate_from_existing_data(self, enhanced):
        """Populate enhanced metadata from existing language data"""
        existing = enhanced
        metadata = enhanced['enhanced_metadata']
        
        # Map existing fields to enhanced structure
        if 'iso639_3' in existing:
            metadata['metadata_sources']['iso639_3'] = existing['iso639_3']
        
        if 'glottocode' in existing:
            metadata['metadata_sources']['glottocode'] = existing['glottocode']
            # Glottolog codes often contain family information
            if existing['glottocode']:
                self._infer_from_glottocode(existing['glottocode'], metadata)
        
        if 'alternative_names' in existing:
            metadata['linguistic_features']['dialects'] = self._parse_alternative_names(existing['alternative_names'])
        
        if 'time_frame' in existing and existing['time_frame']:
            self._parse_time_frame(existing['time_frame'], metadata)
        
        # Infer geographic details from region info
        if 'region_info' in existing:
            self._infer_geographic_details(existing['region_info'], metadata)
    
    def _fetch_external_metadata(self, enhanced):
        """Fetch additional metadata from external sources"""
        metadata = enhanced['enhanced_metadata']
        
        # Try Ethnologue (would need API key)
        if enhanced.get('iso639_3'):
            ethnologue_data = self._fetch_ethnologue_data(enhanced['iso639_3'])
            if ethnologue_data:
                self._integrate_ethnologue_data(ethnologue_data, metadata)
        
        # Try Glottolog (free API)
        if enhanced.get('glottocode'):
            glottolog_data = self._fetch_glottolog_data(enhanced['glottocode'])
            if glottolog_data:
                self._integrate_glottolog_data(glottolog_data, metadata)
        
        # Try WALS (World Atlas of Language Structures)
        wals_data = self._fetch_wals_data(enhanced.get('iso639_3'))
        if wals_data:
            self._integrate_wals_data(wals_data, metadata)
    
    def _infer_from_glottocode(self, glottocode, metadata):
        """Infer information from Glottolog code structure"""
        # Glottolog codes often have hierarchical structure
        # This is a placeholder for more sophisticated inference
        if glottocode.endswith('1234'):  # Example pattern
            metadata['classification']['isolate'] = True
    
    def _parse_alternative_names(self, alt_names):
        """Parse alternative names to extract dialect information"""
        if not alt_names:
            return []
        
        # Simple parsing - could be much more sophisticated
        names = alt_names.split(',') if isinstance(alt_names, str) else alt_names
        return [name.strip() for name in names if name.strip()]
    
    def _parse_time_frame(self, time_frame, metadata):
        """Parse time frame information"""
        # This would need to be adapted based on actual DiACL time_frame structure
        if isinstance(time_frame, dict):
            if 'start' in time_frame:
                metadata['periods']['active_period_start'] = time_frame['start']
            if 'end' in time_frame:
                metadata['periods']['active_period_end'] = time_frame['end']
    
    def _infer_geographic_details(self, region_info, metadata):
        """Infer geographic details from region information"""
        if not region_info:
            return
        
        # Analyze centroids to determine countries/regions
        centroids = []
        total_area = 0
        
        for region in region_info:
            centroids.append((region['centroid_lat'], region['centroid_lon']))
            total_area += region.get('total_area_km2', 0)
        
        # Determine geographic spread based on area and distribution
        if total_area < 1000:  # Less than 1000 km²
            metadata['geographic_details']['geographic_spread'] = 'local'
        elif total_area < 50000:  # Less than 50,000 km²
            metadata['geographic_details']['geographic_spread'] = 'regional'
        else:
            metadata['geographic_details']['geographic_spread'] = 'national'
    
    def _fetch_ethnologue_data(self, iso_code):
        """Fetch data from Ethnologue API (requires API key)"""
        if not self.ethnologue_api_key:
            return None
        
        # Placeholder for Ethnologue API call
        # Would need actual API implementation
        return None
    
    def _fetch_glottolog_data(self, glottocode):
        """Fetch data from Glottolog API (free)"""
        if glottocode in self.glottolog_cache:
            return self.glottolog_cache[glottocode]
        
        try:
            # Glottolog has a JSON API
            url = f"https://glottolog.org/resource/languoid/id/{glottocode}.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.glottolog_cache[glottocode] = data
                return data
        except Exception as e:
            print(f"Error fetching Glottolog data for {glottocode}: {e}")
        
        return None
    
    def _fetch_wals_data(self, iso_code):
        """Fetch typological data from WALS"""
        # WALS doesn't have a direct API, but data is available
        # Would need to implement based on WALS data structure
        return None
    
    def _integrate_ethnologue_data(self, ethnologue_data, metadata):
        """Integrate Ethnologue data into enhanced metadata"""
        # Would need to implement based on Ethnologue API structure
        pass
    
    def _integrate_glottolog_data(self, glottolog_data, metadata):
        """Integrate Glottolog data into enhanced metadata"""
        if 'classification' in glottolog_data:
            # Glottolog provides detailed classification
            classification = glottolog_data['classification']
            metadata['classification']['language_family'] = classification.get('family')
            metadata['classification']['isolate'] = classification.get('isolate', False)
        
        if 'endangerment' in glottolog_data:
            # Glottolog has endangerment information
            endangerment = glottolog_data['endangerment']
            metadata['speakers']['vitality_status'] = endangerment.get('status')
    
    def _integrate_wals_data(self, wals_data, metadata):
        """Integrate WALS typological data"""
        if wals_data:
            metadata['linguistic_features']['typological_features'] = wals_data

def create_enhanced_language_database(cache_file="language_cache.json", output_file="enhanced_language_cache.json"):
    """Create enhanced language database with additional metadata"""
    
    print("🔧 Creating enhanced language database...")
    
    # Load existing cache
    if not Path(cache_file).exists():
        print(f"❌ Cache file {cache_file} not found. Run the basic cache generation first.")
        return None
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    enhancer = LanguageMetadataEnhancer()
    enhanced_cache = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'enhanced_at': datetime.now().isoformat(),
            'enhancement_version': '1.0',
            'total_languages': len(cache_data.get('languages', {})),
        },
        'languages': {},
        'processing_errors': cache_data.get('processing_errors', []),
        'enhancement_errors': []
    }
    
    print(f"Enhancing {len(cache_data['languages'])} languages...")
    
    for i, (lang_id, lang_data) in enumerate(cache_data['languages'].items()):
        try:
            print(f"[{i+1}/{len(cache_data['languages'])}] Enhancing {lang_data['name']}...")
            enhanced_lang = enhancer.enhance_language_data(lang_data)
            enhanced_cache['languages'][lang_id] = enhanced_lang
            
        except Exception as e:
            print(f"❌ Error enhancing {lang_data.get('name', lang_id)}: {e}")
            enhanced_cache['enhancement_errors'].append({
                'language_id': lang_id,
                'name': lang_data.get('name', 'Unknown'),
                'error': str(e)
            })
    
    # Save enhanced cache
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cache, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced database saved to {output_file}")
    print(f"   Languages: {len(enhanced_cache['languages'])}")
    print(f"   Enhancement errors: {len(enhanced_cache['enhancement_errors'])}")
    
    return enhanced_cache

if __name__ == "__main__":
    # Example usage
    enhanced_db = create_enhanced_language_database()
    if enhanced_db:
        print("\n📊 Sample enhanced metadata structure:")
        sample_lang = next(iter(enhanced_db['languages'].values()))
        print(json.dumps(sample_lang['enhanced_metadata'], indent=2))