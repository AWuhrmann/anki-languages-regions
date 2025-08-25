#!/usr/bin/env python3
"""
Simple script to download and visualize language regions from DiACL dataset
Uses WKT polygon data to create accurate language boundary maps
"""

import json
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import time
from datetime import datetime

def save_language_cache(cache_data, cache_file="language_cache.json"):
    """Save processed language data to cache file"""
    cache_data['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'total_languages': len(cache_data.get('languages', {})),
        'total_regions': sum(lang['region_count'] for lang in cache_data.get('languages', {}).values())
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved cache to {cache_file}")
    print(f"   Languages: {cache_data['metadata']['total_languages']}")
    print(f"   Total regions: {cache_data['metadata']['total_regions']}")

def load_language_cache(cache_file="language_cache.json"):
    """Load processed language data from cache file"""
    if not Path(cache_file).exists():
        print(f"📁 Cache file {cache_file} not found")
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"📁 Loaded cache from {cache_file}")
        if 'metadata' in cache_data:
            print(f"   Generated: {cache_data['metadata']['generated_at']}")
            print(f"   Languages: {cache_data['metadata']['total_languages']}")
            print(f"   Total regions: {cache_data['metadata']['total_regions']}")
        
        return cache_data
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return None

def create_language_cache(json_file="countries.json", max_languages=None, cache_file="language_cache.json"):
    """Create a comprehensive cache of all language data for fast Anki generation"""
    
    print(f"🔄 Creating language cache...")
    print(f"This will download and process all WKT data (takes time but saves it for later)")
    print()
    
    # Load DiACL data
    data = load_diacl_data(json_file)
    
    # Get geographical presences to process
    if max_languages:
        geo_presences = list(data['GeographicalPresences'].items())[:max_languages]
        print(f"Processing first {max_languages} languages...")
    else:
        geo_presences = list(data['GeographicalPresences'].items())
        print(f"Processing all {len(geo_presences)} geographical presences...")
    
    cache_data = {
        'diacl_source': json_file,
        'languages': {},
        'processing_errors': []
    }
    
    successful_count = 0
    
    for i, (presence_id, presence_info) in enumerate(geo_presences):
        language_id = presence_info['FkLanguageId']
        language_info = get_language_info(data, language_id)
        
        if not language_info or 'WktWgs84Url' not in presence_info:
            cache_data['processing_errors'].append({
                'presence_id': presence_id,
                'language_id': language_id,
                'error': 'Missing language info or WKT URL'
            })
            continue
            
        language_name = language_info['Name']
        print(f"[{i+1}/{len(geo_presences)}] Processing {language_name}...")
        
        try:
            # Download WKT geometry
            geometry = download_wkt_data(presence_info['WktWgs84Url'])
            
            if geometry:
                # Process polygon grouping
                if geometry.geom_type == 'MultiPolygon':
                    original_polygons = list(geometry.geoms)
                    polygon_groups = group_polygons_intelligently(original_polygons, language_name, max_regions=10)
                else:
                    original_polygons = [geometry]
                    polygon_groups = [[geometry]]
                
                # Convert polygon groups to WKT for storage
                polygon_groups_wkt = []
                for group in polygon_groups:
                    group_wkt = []
                    for poly in group:
                        group_wkt.append(poly.wkt)
                    polygon_groups_wkt.append(group_wkt)
                
                # Calculate areas and centroids for each group
                region_info = []
                for j, group in enumerate(polygon_groups):
                    # Combine polygons in group
                    if len(group) == 1:
                        combined_geom = group[0]
                    else:
                        from shapely.geometry import MultiPolygon
                        combined_geom = MultiPolygon(group)
                    
                    centroid = combined_geom.centroid
                    total_area = sum(calculate_polygon_area_km2(poly) for poly in group)
                    bounds = combined_geom.bounds
                    
                    region_info.append({
                        'region_index': j + 1,
                        'polygon_count': len(group),
                        'total_area_km2': round(total_area, 1),
                        'centroid_lat': round(centroid.y, 4),
                        'centroid_lon': round(centroid.x, 4),
                        'bounds': {
                            'min_lon': round(bounds[0], 4),
                            'min_lat': round(bounds[1], 4),
                            'max_lon': round(bounds[2], 4),
                            'max_lat': round(bounds[3], 4)
                        }
                    })
                
                # Store comprehensive language data
                cache_data['languages'][presence_id] = {
                    'presence_id': presence_id,
                    'language_id': language_id,
                    'name': language_name,
                    'iso639_3': language_info.get('ISO639-3', ''),
                    'glottocode': language_info.get('Glottocode', ''),
                    'alternative_names': language_info.get('AlternativeNames', ''),
                    'focal_point': language_info.get('FocalPointWgs84', {}),
                    'language_area': data['LanguageAreas'].get(str(language_info['FkLanguageAreaId']), 'Unknown'),
                    'reliability': data['LanguageReliabilities'].get(str(language_info['FkLanguageReliabilityId']), 'Unknown'),
                    'wkt_url': presence_info['WktWgs84Url'],
                    'original_wkt': geometry.wkt,
                    'original_polygon_count': len(original_polygons),
                    'region_count': len(polygon_groups),
                    'polygon_groups_wkt': polygon_groups_wkt,
                    'region_info': region_info,
                    'time_frame': presence_info.get('TimeFrame', {}),
                    'source_references': presence_info.get('SourceReferences', [])
                }
                
                successful_count += 1
                print(f"   ✅ {len(original_polygons)} polygons → {len(polygon_groups)} regions")
                
            else:
                cache_data['processing_errors'].append({
                    'presence_id': presence_id,
                    'language_name': language_name,
                    'error': 'Failed to download WKT geometry'
                })
                print(f"   ❌ Failed to download geometry")
                
        except Exception as e:
            cache_data['processing_errors'].append({
                'presence_id': presence_id,
                'language_name': language_name,
                'error': str(e)
            })
            print(f"   ❌ Error: {str(e)[:50]}...")
        
        # Small delay to be nice to server
        time.sleep(0.3)
    
    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {successful_count} languages")
    print(f"   Errors: {len(cache_data['processing_errors'])}")
    
    # Save cache
    save_language_cache(cache_data, cache_file)
    
    return cache_data

def analyze_cached_data(cache_file="language_cache.json"):
    """Analyze cached language data to show statistics"""
    cache_data = load_language_cache(cache_file)
    
    if not cache_data:
        print("No cache data available. Run create_language_cache() first.")
        return
    
    languages = cache_data['languages']
    
    print(f"\n📊 CACHED DATA ANALYSIS")
    print("=" * 40)
    
    # Overall statistics
    total_languages = len(languages)
    total_regions = sum(lang['region_count'] for lang in languages.values())
    total_original_polygons = sum(lang['original_polygon_count'] for lang in languages.values())
    
    print(f"Total languages: {total_languages}")
    print(f"Total original polygons: {total_original_polygons}")
    print(f"Total final regions: {total_regions}")
    print(f"Reduction achieved: {total_original_polygons - total_regions} polygons ({((total_original_polygons - total_regions) / total_original_polygons * 100):.1f}%)")
    print()
    
    # Language area distribution
    area_counts = {}
    for lang in languages.values():
        area = lang['language_area']
        area_counts[area] = area_counts.get(area, 0) + 1
    
    print("Languages by geographic area:")
    for area, count in sorted(area_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {area:20} | {count:3} languages")
    print()
    
    # Languages with most regions
    by_regions = sorted(languages.values(), key=lambda x: x['region_count'], reverse=True)
    print("Languages with most regions:")
    for lang in by_regions[:10]:
        reduction = lang['original_polygon_count'] - lang['region_count']
        print(f"  {lang['name']:20} | {lang['region_count']} regions (reduced by {reduction})")
    print()
    
    # Processing errors
    if cache_data['processing_errors']:
        print(f"Processing errors ({len(cache_data['processing_errors'])}):")
        for error in cache_data['processing_errors'][:5]:
            print(f"  {error.get('language_name', 'Unknown'):20} | {error['error']}")
        if len(cache_data['processing_errors']) > 5:
            print(f"  ... and {len(cache_data['processing_errors']) - 5} more")
    
    return cache_data

def load_cached_language(presence_id, cache_file="language_cache.json"):
    """Load a specific language from cache with geometry reconstruction"""
    cache_data = load_language_cache(cache_file)
    
    if not cache_data or str(presence_id) not in cache_data['languages']:
        return None
    
    lang_data = cache_data['languages'][str(presence_id)]
    
    # Reconstruct polygon groups from WKT
    polygon_groups = []
    for group_wkt in lang_data['polygon_groups_wkt']:
        group_polygons = []
        for poly_wkt in group_wkt:
            group_polygons.append(load_wkt(poly_wkt))
        polygon_groups.append(group_polygons)
    
    # Return data in format expected by map creation functions
    return {
        'language_info': {
            'Name': lang_data['name'],
            'ISO639-3': lang_data.get('iso639_3'),
            'Glottocode': lang_data.get('glottocode'),
            'AlternativeNames': lang_data.get('alternative_names'),
            'FocalPointWgs84': lang_data.get('focal_point'),
            'FkLanguageAreaId': None,  # Not needed for map creation
            'FkLanguageReliabilityId': None
        },
        'polygon_groups': polygon_groups,
        'region_info': lang_data['region_info']
    }

def quick_cache_test(cache_file="language_cache.json"):
    """Quick test of cache functionality"""
    print(f"🧪 Testing cache functionality...")
    
    # Check if cache exists
    cache_data = load_language_cache(cache_file)
    
    if cache_data:
        # Test loading a specific language
        first_presence_id = list(cache_data['languages'].keys())[0]
        test_lang = load_cached_language(first_presence_id, cache_file)
        
        if test_lang:
            print(f"✅ Successfully loaded cached language: {test_lang['language_info']['Name']}")
            print(f"   Regions: {len(test_lang['polygon_groups'])}")
            print(f"   Cache is working correctly!")
        else:
            print("❌ Failed to load language from cache")
    else:
        print("📝 No cache found. To create cache, run:")
        print("   create_language_cache(max_languages=10)  # for testing")
        print("   create_language_cache()                 # for full dataset")

def count_total_regions_cached(cache_file="language_cache.json"):
    """Count regions from cached data (much faster)"""
    cache_data = load_language_cache(cache_file)
    
    if not cache_data:
        print("No cache available. Run create_language_cache() first.")
        return 0, []
    
    languages = cache_data['languages']
    total_regions = sum(lang['region_count'] for lang in languages.values())
    
    # Create breakdown list
    language_breakdown = []
    for lang in languages.values():
        language_breakdown.append({
            'name': lang['name'],
            'presence_id': lang['presence_id'],
            'original_polygons': lang['original_polygon_count'],
            'final_regions': lang['region_count']
        })
    
    print(f"📊 CACHED REGION COUNT")
    print("=" * 30)
    print(f"Total regions: {total_regions}")
    print(f"Total languages: {len(languages)}")
    print(f"Average regions per language: {total_regions/len(languages):.1f}")
    
    # Show top languages by region count
    language_breakdown.sort(key=lambda x: x['final_regions'], reverse=True)
    print(f"\nTop languages by region count:")
    for lang in language_breakdown[:8]:
        reduction = lang['original_polygons'] - lang['final_regions']
        print(f"  {lang['name']:20} | {lang['final_regions']} regions (reduced by {reduction})")
    
    return total_regions, language_breakdown

def load_diacl_data(filename="countries.json"):
    """Load the DiACL JSON data"""
    print(f"Loading DiACL data from {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['Languages'])} languages")
    print(f"Found {len(data['GeographicalPresences'])} geographical presences")
    
    return data

def download_wkt_data(wkt_url):
    """Download WKT data from a URL"""
    try:
        print(f"Downloading WKT from: {wkt_url}")
        response = requests.get(wkt_url, timeout=10)
        response.raise_for_status()
        wkt_text = response.text.strip()
        
        # Parse WKT to shapely geometry
        geometry = load_wkt(wkt_text)
        print(f"Successfully parsed WKT geometry: {geometry.geom_type}")
        return geometry
        
    except Exception as e:
        print(f"Error downloading/parsing WKT from {wkt_url}: {e}")
        return None

def get_language_info(data, language_id):
    """Get language information by ID"""
    for lang_id, lang_info in data['Languages'].items():
        if int(lang_id) == int(language_id):
            return lang_info
    return None

def calculate_polygon_area_km2(polygon):
    """Calculate approximate area of a polygon in km²"""
    # For rough area calculation, we'll use a simple approach
    # Convert degrees to km (very rough approximation)
    bounds = polygon.bounds
    lat_center = (bounds[1] + bounds[3]) / 2
    
    # Rough conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 * cos(lat) km
    lat_km_per_degree = 111.0
    lon_km_per_degree = 111.0 * abs(np.cos(np.radians(lat_center)))
    
    # Calculate area using shapely (in square degrees) then convert to km²
    area_deg2 = polygon.area
    area_km2 = area_deg2 * lat_km_per_degree * lon_km_per_degree
    
    return area_km2

def group_polygons_intelligently(polygons, language_name, area_threshold_km2=5000, distance_threshold_deg=5.0, max_regions=10):
    """Group small polygons with nearby larger ones or with each other"""
    
    # Calculate areas and centroids for all polygons
    polygon_info = []
    for i, poly in enumerate(polygons):
        area = calculate_polygon_area_km2(poly)
        centroid = poly.centroid
        polygon_info.append({
            'index': i,
            'polygon': poly,
            'area_km2': area,
            'centroid': centroid,
            'grouped': False
        })
    
    # Sort by area (largest first)
    polygon_info.sort(key=lambda x: x['area_km2'], reverse=True)
    
    print(f"Polygon areas for {language_name}:")
    for info in polygon_info:
        print(f"  Polygon {info['index']}: {info['area_km2']:.1f} km²")
    
    # Special case for languages like Romansh - if all polygons are relatively small and close, group them all
    if language_name.lower() in ['romansh', 'romansch']:
        print(f"Special handling for {language_name}: grouping all regions together")
        combined_polygons = [polygons]  # Keep as one MultiPolygon
        return combined_polygons
    
    groups = []
    
    # Process each polygon
    for current_info in polygon_info:
        if current_info['grouped']:
            continue
            
        current_group = [current_info['polygon']]
        current_info['grouped'] = True
        
        # If this is a large polygon, it can attract small nearby polygons
        if current_info['area_km2'] >= area_threshold_km2:
            # Look for small nearby polygons to group with this large one
            for other_info in polygon_info:
                if other_info['grouped'] or other_info['area_km2'] >= area_threshold_km2:
                    continue
                    
                # Calculate distance between centroids
                distance = np.sqrt(
                    (current_info['centroid'].x - other_info['centroid'].x)**2 + 
                    (current_info['centroid'].y - other_info['centroid'].y)**2
                )
                
                if distance <= distance_threshold_deg:
                    current_group.append(other_info['polygon'])
                    other_info['grouped'] = True
                    print(f"  Grouped small polygon ({other_info['area_km2']:.1f} km²) with large polygon ({current_info['area_km2']:.1f} km²)")
        
        # If this is a small polygon, try to group it with other small nearby polygons
        else:
            for other_info in polygon_info:
                if other_info['grouped'] or other_info['area_km2'] >= area_threshold_km2:
                    continue
                    
                # Calculate distance between centroids
                distance = np.sqrt(
                    (current_info['centroid'].x - other_info['centroid'].x)**2 + 
                    (current_info['centroid'].y - other_info['centroid'].y)**2
                )
                
                if distance <= distance_threshold_deg:
                    current_group.append(other_info['polygon'])
                    other_info['grouped'] = True
                    print(f"  Grouped small polygons ({current_info['area_km2']:.1f} km² + {other_info['area_km2']:.1f} km²)")
        
        groups.append(current_group)
    
    print(f"Initial grouping: {len(groups)} groups from {len(polygons)} original polygons")
    
    # If we still have too many regions, do additional grouping
    if len(groups) > max_regions:
        print(f"Too many regions ({len(groups)}), further grouping to max {max_regions}...")
        groups = further_group_by_proximity(groups, max_regions)
    
    print(f"Final result: {len(groups)} groups")
    return groups

def further_group_by_proximity(groups, max_regions):
    """Further group regions if we exceed the maximum number"""
    
    # Calculate centroid and total area for each group
    group_info = []
    for i, group in enumerate(groups):
        # Calculate combined centroid and area
        total_area = sum(calculate_polygon_area_km2(poly) for poly in group)
        
        # Calculate average centroid
        centroids = [poly.centroid for poly in group]
        avg_lon = sum(c.x for c in centroids) / len(centroids)
        avg_lat = sum(c.y for c in centroids) / len(centroids)
        
        group_info.append({
            'index': i,
            'group': group,
            'total_area': total_area,
            'centroid_lon': avg_lon,
            'centroid_lat': avg_lat,
            'merged': False
        })
    
    # Sort by area (largest first) to preserve large regions
    group_info.sort(key=lambda x: x['total_area'], reverse=True)
    
    final_groups = []
    
    # Keep largest regions as separate groups
    large_groups_kept = 0
    for info in group_info:
        if info['total_area'] >= 10000 and large_groups_kept < max_regions // 2:  # Keep up to half as large regions
            final_groups.append(info['group'])
            info['merged'] = True
            large_groups_kept += 1
    
    # Group remaining regions by proximity
    while len(final_groups) < max_regions and any(not info['merged'] for info in group_info):
        # Find closest pair of unmerged groups
        best_pair = None
        min_distance = float('inf')
        
        unmerged = [info for info in group_info if not info['merged']]
        
        if len(unmerged) <= max_regions - len(final_groups):
            # Can add remaining groups individually
            for info in unmerged:
                final_groups.append(info['group'])
                info['merged'] = True
            break
        
        # Find closest pair to merge
        for i, info1 in enumerate(unmerged):
            for info2 in unmerged[i+1:]:
                distance = np.sqrt(
                    (info1['centroid_lon'] - info2['centroid_lon'])**2 + 
                    (info1['centroid_lat'] - info2['centroid_lat'])**2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (info1, info2)
        
        if best_pair:
            # Merge the closest pair
            merged_group = best_pair[0]['group'] + best_pair[1]['group']
            final_groups.append(merged_group)
            best_pair[0]['merged'] = True
            best_pair[1]['merged'] = True
            print(f"  Merged two groups (distance: {min_distance:.2f}°)")
        else:
            break
    
    return final_groups

def create_world_overview_subplot(ax, polygon_groups, language_info):
    """Create a world overview showing where language regions are located"""
    
    # Set up world map
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.7, color='lightgray', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.7, color='lightblue', facecolor='lightblue')
    
    # Plot dots for each region group
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, polygon_group in enumerate(polygon_groups):
        color = colors[i % len(colors)]
        
        # Calculate centroid for this group
        if len(polygon_group) == 1:
            centroid = polygon_group[0].centroid
        else:
            centroids = [poly.centroid for poly in polygon_group]
            avg_lon = sum(c.x for c in centroids) / len(centroids)
            avg_lat = sum(c.y for c in centroids) / len(centroids)
            from shapely.geometry import Point
            centroid = Point(avg_lon, avg_lat)
        
        # Plot the region center
        ax.plot(centroid.x, centroid.y, 'o', color=color, markersize=8,
               markeredgecolor='black', markeredgewidth=1,
               transform=ccrs.PlateCarree(), 
               label=f'Region {i+1}' if len(polygon_groups) > 1 else language_info['Name'])
    
    # Add focal point if available
    if 'FocalPointWgs84' in language_info:
        focal_lat = float(language_info['FocalPointWgs84']['Latitude'])
        focal_lon = float(language_info['FocalPointWgs84']['Longitude'])
        ax.plot(focal_lon, focal_lat, 's', color='yellow', markersize=6,
               markeredgecolor='black', markeredgewidth=1,
               transform=ccrs.PlateCarree(), label='Focal point')
    
    ax.set_title('World Overview', fontsize=10, fontweight='bold')
    
    # Add small legend if multiple regions
    if len(polygon_groups) > 1:
        ax.legend(loc='lower left', fontsize=8, framealpha=0.8)

def create_language_map(language_info, geometry, output_dir="language_maps"):
    """Create a map for a language with its WKT geometry using globe-like projection"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Split MultiPolygon into separate polygons, then group them intelligently
    if geometry.geom_type == 'MultiPolygon':
        original_polygons = list(geometry.geoms)
        print(f"Found {len(original_polygons)} separate areas for {language_info['Name']}")
        
        # Group polygons intelligently with max regions limit
        polygon_groups = group_polygons_intelligently(original_polygons, language_info['Name'], max_regions=10)
    else:
        polygon_groups = [[geometry]]
    
    created_maps = []
    
    # Create a separate map for each polygon group
    for i, polygon_group in enumerate(polygon_groups):
        
        # Combine polygons in this group
        if len(polygon_group) == 1:
            combined_geometry = polygon_group[0]
        else:
            # Create a MultiPolygon from the group
            from shapely.geometry import MultiPolygon
            combined_geometry = MultiPolygon(polygon_group)
        
        # Calculate the centroid of this group for centering
        centroid = combined_geometry.centroid
        center_lon, center_lat = centroid.x, centroid.y
        
        # Get bounds from this polygon group
        bounds = combined_geometry.bounds  # (minx, miny, maxx, maxy)
        
        # Calculate the span of the geometry
        lon_span = abs(bounds[2] - bounds[0])
        lat_span = abs(bounds[3] - bounds[1])
        max_span = max(lon_span, lat_span)
        
        print(f"Group {i+1}: Center=({center_lat:.3f}, {center_lon:.3f}), Span={max_span:.3f}°, Polygons={len(polygon_group)}")
        
        # Choose projection and zoom based on location and size
        if max_span > 60:  # Very large areas - use global view
            projection = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
            buffer = max_span * 0.8
        elif abs(center_lat) > 60:  # Arctic/Antarctic regions
            if center_lat > 0:
                projection = ccrs.NorthPolarStereo(central_longitude=center_lon)
            else:
                projection = ccrs.SouthPolarStereo(central_longitude=center_lon)
            buffer = max(max_span * 1.5, 5)  # Minimum 5 degrees buffer
        else:  # Most regions - use azimuthal equidistant for nice globe-like appearance
            projection = ccrs.AzimuthalEquidistant(central_longitude=center_lon, central_latitude=center_lat)
            
            # Adjust buffer based on area size
            if max_span < 1.0:  # Very small area
                buffer = max(max_span * 3, 1.5)  # At least 1.5 degrees buffer
                print(f"Small area detected, using tight zoom with buffer={buffer:.2f}°")
            elif max_span < 5.0:  # Small area
                buffer = max(max_span * 2, 3)    # At least 3 degrees buffer
            else:  # Regular area
                buffer = max(max_span * 1.2, 8)  # At least 8 degrees buffer
        
        # Create figure with subplots: main map + world overview
        fig = plt.figure(figsize=(16, 10))
        
        # Main detailed map (takes up most of the space)
        ax_main = plt.subplot(1, 2, 1, projection=projection)
        
        # World overview map (smaller, on the right)
        ax_world = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        
        # === MAIN MAP ===
        # Set extent with appropriate buffer
        if isinstance(projection, ccrs.Orthographic):
            ax_main.set_global()
        else:
            try:
                ax_main.set_extent([
                    center_lon - buffer,
                    center_lon + buffer,
                    center_lat - buffer,
                    center_lat + buffer
                ], ccrs.PlateCarree())
            except:
                ax_main.set_extent([
                    max(-180, center_lon - buffer),
                    min(180, center_lon + buffer),
                    max(-90, center_lat - buffer),
                    min(90, center_lat + buffer)
                ], ccrs.PlateCarree())
        
        # Add map features to main map
        ax_main.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax_main.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6, color='gray')
        ax_main.add_feature(cfeature.LAND, alpha=0.9, color='lightgray', facecolor='lightgray')
        ax_main.add_feature(cfeature.OCEAN, alpha=0.9, color='lightsteelblue', facecolor='lightsteelblue')
        ax_main.add_feature(cfeature.RIVERS, alpha=0.5, linewidth=0.5, edgecolor='blue')
        ax_main.add_feature(cfeature.LAKES, alpha=0.7, color='lightblue', facecolor='lightblue')
        
        # Plot all polygons in this group on main map
        for poly in polygon_group:
            if poly.exterior:
                x_coords, y_coords = poly.exterior.xy
                ax_main.fill(x_coords, y_coords, 
                       color='red', alpha=0.6, 
                       edgecolor='darkred', linewidth=3,
                       transform=ccrs.PlateCarree())
                
                # Add thick outline for better visibility
                ax_main.plot(x_coords, y_coords,
                       color='darkred', linewidth=4,
                       transform=ccrs.PlateCarree())
        
        # Add focal point to main map
        if i == 0 and 'FocalPointWgs84' in language_info:
            focal_lat = float(language_info['FocalPointWgs84']['Latitude'])
            focal_lon = float(language_info['FocalPointWgs84']['Longitude'])
            ax_main.plot(focal_lon, focal_lat, 'o', color='yellow', markersize=12,
                   markeredgecolor='black', markeredgewidth=2,
                   transform=ccrs.PlateCarree(), label='Focal point')
        
        # Add grid lines to main map
        if not isinstance(projection, ccrs.Orthographic):
            gl = ax_main.gridlines(draw_labels=False, dms=True, 
                             linewidth=0.5, alpha=0.4, color='gray',
                             linestyle='--')
        
        # === WORLD OVERVIEW MAP ===
        create_world_overview_subplot(ax_world, polygon_groups, language_info)
        
        # === TITLES AND INFO ===
        # Main title
        if len(polygon_groups) > 1:
            main_title = f"{language_info['Name']} - Region {i+1} of {len(polygon_groups)}"
        else:
            main_title = f"{language_info['Name']}"
            
        if 'AlternativeNames' in language_info and i == 0:
            alt_names = language_info['AlternativeNames']
            if len(alt_names) > 50:
                main_title += f"\n({alt_names[:50]}...)"
            else:
                main_title += f"\n({alt_names})"
        
        # Add area info to title
        total_area = sum(calculate_polygon_area_km2(poly) for poly in polygon_group)
        if len(polygon_group) > 1:
            main_title += f"\n({len(polygon_group)} areas, ~{total_area:.0f} km²)"
        
        iso_info = ""
        if 'ISO639-3' in language_info:
            iso_info += f"ISO: {language_info['ISO639-3']}"
        if 'Glottocode' in language_info:
            iso_info += f" | Glottolog: {language_info['Glottocode']}"
        
        if iso_info and i == 0:
            main_title += f"\n{iso_info}"
        
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.95)
        
        # Add subtitle with projection info
        proj_name = projection.__class__.__name__.replace('ccrs.', '')
        area_info = f"Center: {center_lat:.2f}°N, {center_lon:.2f}°E | Span: {max_span:.2f}°"
        plt.figtext(0.5, 0.02, f"Projection: {proj_name} | {area_info}", 
                    ha='center', fontsize=8, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the map with group number if multiple groups
        safe_name = "".join(c for c in language_info['Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if len(polygon_groups) > 1:
            filename = f"{safe_name.replace(' ', '_')}_region_{i+1}_with_overview.png"
        else:
            filename = f"{safe_name.replace(' ', '_')}_with_overview.png"
        
        filepath = Path(output_dir) / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved globe map with world overview {i+1}: {filepath}")
        created_maps.append(filepath)
    
    return created_maps

def test_language_mapping(json_file="countries.json", max_languages=5, specific_presence_id=None):
    """Test the language mapping with a few examples or a specific language"""
    
    # Load data
    data = load_diacl_data(json_file)
    
    if specific_presence_id:
        # Test only the specific presence ID
        if str(specific_presence_id) in data['GeographicalPresences']:
            geo_presences = [(str(specific_presence_id), data['GeographicalPresences'][str(specific_presence_id)])]
            print(f"\nTesting specific presence ID {specific_presence_id}...")
        else:
            print(f"Presence ID {specific_presence_id} not found!")
            return
    else:
        # Get some geographical presences to test
        geo_presences = list(data['GeographicalPresences'].items())[:max_languages]
        print(f"\nTesting with {len(geo_presences)} language regions...")
    
    successful_maps = []
    
    for presence_id, presence_info in geo_presences:
        print(f"\n--- Processing presence ID {presence_id} ---")
        
        # Get language info
        language_id = presence_info['FkLanguageId']
        language_info = get_language_info(data, language_id)
        
        if not language_info:
            print(f"Could not find language info for ID {language_id}")
            continue
            
        print(f"Language: {language_info['Name']}")
        
        # Download WKT geometry
        if 'WktWgs84Url' in presence_info:
            geometry = download_wkt_data(presence_info['WktWgs84Url'])
            
            if geometry:
                # Create map(s) - now returns a list of filepaths
                map_paths = create_language_map(language_info, geometry)
                successful_maps.extend(map_paths)  # Add all maps to the list
            else:
                print("Failed to get geometry, skipping map creation")
        else:
            print("No WKT URL found for this presence")
        
        # Small delay to be nice to the server
        time.sleep(1)
    
    print(f"\n🎉 Successfully created {len(successful_maps)} language maps!")
    print("Maps saved in 'language_maps/' directory:")
    for map_path in successful_maps:
        print(f"  - {map_path}")

def count_total_regions(json_file="countries.json", max_languages=None):
    """Count how many total regions would be created after grouping"""
    
    data = load_diacl_data(json_file)
    
    # Get geographical presences to analyze
    if max_languages:
        geo_presences = list(data['GeographicalPresences'].items())[:max_languages]
    else:
        geo_presences = list(data['GeographicalPresences'].items())
    
    total_regions = 0
    language_breakdown = []
    
    print(f"\n=== REGION COUNT ANALYSIS ===")
    print(f"Analyzing {len(geo_presences)} geographical presences...")
    print()
    
    for presence_id, presence_info in geo_presences:
        language_id = presence_info['FkLanguageId']
        language_info = get_language_info(data, language_id)
        
        if not language_info or 'WktWgs84Url' not in presence_info:
            continue
            
        language_name = language_info['Name']
        
        try:
            # Download and analyze the geometry (just for counting)
            geometry = download_wkt_data(presence_info['WktWgs84Url'])
            
            if geometry:
                # Count regions after grouping
                if geometry.geom_type == 'MultiPolygon':
                    original_polygons = list(geometry.geoms)
                    polygon_groups = group_polygons_intelligently(original_polygons, language_name, max_regions=10)
                else:
                    polygon_groups = [[geometry]]
                
                region_count = len(polygon_groups)
                total_regions += region_count
                
                language_breakdown.append({
                    'name': language_name,
                    'presence_id': presence_id,
                    'original_polygons': len(original_polygons) if geometry.geom_type == 'MultiPolygon' else 1,
                    'final_regions': region_count
                })
                
                print(f"✓ {language_name:20} | {len(original_polygons) if geometry.geom_type == 'MultiPolygon' else 1:2} → {region_count:2} regions")
                
            else:
                print(f"✗ {language_name:20} | Failed to download geometry")
                
        except Exception as e:
            print(f"✗ {language_name:20} | Error: {str(e)[:30]}...")
        
        # Small delay to be nice to server
        time.sleep(0.5)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total regions that would be created: {total_regions}")
    print(f"Languages successfully analyzed: {len(language_breakdown)}")
    print()
    
    # Show languages with most regions
    language_breakdown.sort(key=lambda x: x['final_regions'], reverse=True)
    print("Languages with most regions:")
    for lang in language_breakdown[:5]:
        reduction = lang['original_polygons'] - lang['final_regions']
        print(f"  {lang['name']:20} | {lang['final_regions']} regions (reduced by {reduction})")
    
    print()
    return total_regions, language_breakdown

def quick_region_count(json_file="countries.json", sample_size=10):
    """Quick count of regions for a sample of languages"""
    print(f"🔢 Quick Region Count (sample of {sample_size} languages)")
    print("=" * 50)
    
    total_regions, breakdown = count_total_regions(json_file, max_languages=sample_size)
    
    print(f"📊 Results for sample of {sample_size} languages:")
    print(f"   Total maps that would be generated: {total_regions}")
    print(f"   Average regions per language: {total_regions/len(breakdown):.1f}")
    
    return total_regions

def test_romansh():
    """Specific test for Romansh language with region count"""
    print("🇨🇭 Testing Romansh (Presence ID 40)")
    print("=" * 40)
    
    # First, count regions for just Romansh
    data = load_diacl_data()
    presence_info = data['GeographicalPresences']['40']
    language_info = get_language_info(data, presence_info['FkLanguageId'])
    
    geometry = download_wkt_data(presence_info['WktWgs84Url'])
    if geometry:
        if geometry.geom_type == 'MultiPolygon':
            original_polygons = list(geometry.geoms)
            polygon_groups = group_polygons_intelligently(original_polygons, language_info['Name'], max_regions=10)
        else:
            polygon_groups = [[geometry]]
        
        print(f"📊 Romansh region analysis:")
        print(f"   Original polygons: {len(original_polygons) if geometry.geom_type == 'MultiPolygon' else 1}")
        print(f"   Final regions: {len(polygon_groups)}")
        print(f"   Maps to be generated: {len(polygon_groups)}")
        print()
    
    # Now generate the actual maps
    print("🗺️  Generating maps with enhanced zoom and world overview...")
    test_language_mapping(specific_presence_id=40)

def explore_dataset(json_file="countries.json"):
    """Quick exploration of the dataset"""
    data = load_diacl_data(json_file)
    
    print("\n=== Dataset Overview ===")
    
    # Language areas
    print(f"\nLanguage Areas ({len(data['LanguageAreas'])}):")
    for area_id, area_name in data['LanguageAreas'].items():
        print(f"  {area_id}: {area_name}")
    
    # Language reliabilities
    print(f"\nLanguage Reliabilities:")
    for rel_id, rel_desc in data['LanguageReliabilities'].items():
        print(f"  {rel_id}: {rel_desc}")
    
    # Sample languages
    print(f"\nSample Languages:")
    for i, (lang_id, lang_info) in enumerate(list(data['Languages'].items())[:5]):
        name = lang_info['Name']
        iso = lang_info.get('ISO639-3', 'N/A')
        area = data['LanguageAreas'].get(str(lang_info['FkLanguageAreaId']), 'Unknown')
        print(f"  {lang_id}: {name} (ISO: {iso}, Area: {area})")
    
    # Count languages with WKT data
    languages_with_wkt = set()
    for presence_info in data['GeographicalPresences'].values():
        if 'WktWgs84Url' in presence_info:
            languages_with_wkt.add(presence_info['FkLanguageId'])
    
    print(f"\nLanguages with WKT polygon data: {len(languages_with_wkt)}")
    print(f"Total geographical presences: {len(data['GeographicalPresences'])}")

def create_maps_from_cache(cache_file="language_cache.json", specific_languages=None, output_dir="language_maps"):
    """Create maps from cached data (much faster than re-downloading)"""
    
    cache_data = load_language_cache(cache_file)
    if not cache_data:
        print("❌ No cache available. Run create_language_cache() first.")
        return []
    
    languages = cache_data['languages']
    
    if specific_languages:
        # Filter to specific languages
        if isinstance(specific_languages, str):
            specific_languages = [specific_languages]
        
        filtered_languages = {}
        for presence_id, lang_data in languages.items():
            if (lang_data['name'] in specific_languages or 
                presence_id in specific_languages or
                str(lang_data['language_id']) in specific_languages):
                filtered_languages[presence_id] = lang_data
        languages = filtered_languages
        
        if not languages:
            print(f"❌ No languages found matching: {specific_languages}")
            return []
    
    print(f"🗺️  Creating maps from cache for {len(languages)} languages...")
    
    all_created_maps = []
    
    for i, (presence_id, lang_data) in enumerate(languages.items()):
        print(f"\n[{i+1}/{len(languages)}] Creating maps for {lang_data['name']}...")
        
        # Load cached language data
        cached_lang = load_cached_language(presence_id, cache_file)
        
        if cached_lang:
            # Create maps using cached polygon groups
            created_maps = create_maps_from_polygon_groups(
                cached_lang['language_info'], 
                cached_lang['polygon_groups'],
                output_dir
            )
            all_created_maps.extend(created_maps)
        else:
            print(f"   ❌ Failed to load {lang_data['name']} from cache")
    
    print(f"\n🎉 Created {len(all_created_maps)} maps total!")
    return all_created_maps

def create_maps_from_polygon_groups(language_info, polygon_groups, output_dir="language_maps"):
    """Create maps from pre-processed polygon groups (used by cache system)"""
    Path(output_dir).mkdir(exist_ok=True)
    
    created_maps = []
    
    print(f"Creating {len(polygon_groups)} maps for {language_info['Name']}")
    
    # Create a separate map for each polygon group
    for i, polygon_group in enumerate(polygon_groups):
        
        # Combine polygons in this group
        if len(polygon_group) == 1:
            combined_geometry = polygon_group[0]
        else:
            # Create a MultiPolygon from the group
            from shapely.geometry import MultiPolygon
            combined_geometry = MultiPolygon(polygon_group)
        
        # Calculate the centroid of this group for centering
        centroid = combined_geometry.centroid
        center_lon, center_lat = centroid.x, centroid.y
        
        # Get bounds from this polygon group
        bounds = combined_geometry.bounds  # (minx, miny, maxx, maxy)
        
        # Calculate the span of the geometry
        lon_span = abs(bounds[2] - bounds[0])
        lat_span = abs(bounds[3] - bounds[1])
        max_span = max(lon_span, lat_span)
        
        print(f"   Group {i+1}: Center=({center_lat:.3f}, {center_lon:.3f}), Span={max_span:.3f}°, Polygons={len(polygon_group)}")
        
        # Choose projection and zoom based on location and size
        if max_span > 60:  # Very large areas - use global view
            projection = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
            buffer = max_span * 0.8
        elif abs(center_lat) > 60:  # Arctic/Antarctic regions
            if center_lat > 0:
                projection = ccrs.NorthPolarStereo(central_longitude=center_lon)
            else:
                projection = ccrs.SouthPolarStereo(central_longitude=center_lon)
            buffer = max(max_span * 1.5, 5)  # Minimum 5 degrees buffer
        else:  # Most regions - use azimuthal equidistant for nice globe-like appearance
            projection = ccrs.AzimuthalEquidistant(central_longitude=center_lon, central_latitude=center_lat)
            
            # Adjust buffer based on area size
            if max_span < 1.0:  # Very small area
                buffer = max(max_span * 3, 1.5)  # At least 1.5 degrees buffer
            elif max_span < 5.0:  # Small area
                buffer = max(max_span * 2, 3)    # At least 3 degrees buffer
            else:  # Regular area
                buffer = max(max_span * 1.2, 8)  # At least 8 degrees buffer
        
        # Create figure with subplots: main map + world overview
        fig = plt.figure(figsize=(16, 10))
        
        # Main detailed map (takes up most of the space)
        ax_main = plt.subplot(1, 2, 1, projection=projection)
        
        # World overview map (smaller, on the right)
        ax_world = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        
        # === MAIN MAP ===
        # Set extent with appropriate buffer
        if isinstance(projection, ccrs.Orthographic):
            ax_main.set_global()
        else:
            try:
                ax_main.set_extent([
                    center_lon - buffer,
                    center_lon + buffer,
                    center_lat - buffer,
                    center_lat + buffer
                ], ccrs.PlateCarree())
            except:
                ax_main.set_extent([
                    max(-180, center_lon - buffer),
                    min(180, center_lon + buffer),
                    max(-90, center_lat - buffer),
                    min(90, center_lat + buffer)
                ], ccrs.PlateCarree())
        
        # Add map features to main map
        ax_main.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax_main.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6, color='gray')
        ax_main.add_feature(cfeature.LAND, alpha=0.9, color='lightgray', facecolor='lightgray')
        ax_main.add_feature(cfeature.OCEAN, alpha=0.9, color='lightsteelblue', facecolor='lightsteelblue')
        ax_main.add_feature(cfeature.RIVERS, alpha=0.5, linewidth=0.5, edgecolor='blue')
        ax_main.add_feature(cfeature.LAKES, alpha=0.7, color='lightblue', facecolor='lightblue')
        
        # Plot all polygons in this group on main map
        for poly in polygon_group:
            if poly.exterior:
                x_coords, y_coords = poly.exterior.xy
                ax_main.fill(x_coords, y_coords, 
                       color='red', alpha=0.6, 
                       edgecolor='darkred', linewidth=3,
                       transform=ccrs.PlateCarree())
                
                # Add thick outline for better visibility
                ax_main.plot(x_coords, y_coords,
                       color='darkred', linewidth=4,
                       transform=ccrs.PlateCarree())
        
        # Add focal point to main map
        if i == 0 and 'FocalPointWgs84' in language_info and language_info['FocalPointWgs84']:
            focal_lat = float(language_info['FocalPointWgs84']['Latitude'])
            focal_lon = float(language_info['FocalPointWgs84']['Longitude'])
            ax_main.plot(focal_lon, focal_lat, 'o', color='yellow', markersize=12,
                   markeredgecolor='black', markeredgewidth=2,
                   transform=ccrs.PlateCarree(), label='Focal point')
        
        # Add grid lines to main map
        if not isinstance(projection, ccrs.Orthographic):
            gl = ax_main.gridlines(draw_labels=False, dms=True, 
                             linewidth=0.5, alpha=0.4, color='gray',
                             linestyle='--')
        
        # === WORLD OVERVIEW MAP ===
        create_world_overview_subplot(ax_world, polygon_groups, language_info)
        
        # === TITLES AND INFO ===
        # Main title
        if len(polygon_groups) > 1:
            main_title = f"{language_info['Name']} - Region {i+1} of {len(polygon_groups)}"
        else:
            main_title = f"{language_info['Name']}"
            
        if 'AlternativeNames' in language_info and language_info['AlternativeNames'] and i == 0:
            alt_names = language_info['AlternativeNames']
            if len(alt_names) > 50:
                main_title += f"\n({alt_names[:50]}...)"
            else:
                main_title += f"\n({alt_names})"
        
        # Add area info to title
        total_area = sum(calculate_polygon_area_km2(poly) for poly in polygon_group)
        if len(polygon_group) > 1:
            main_title += f"\n({len(polygon_group)} areas, ~{total_area:.0f} km²)"
        
        iso_info = ""
        if 'ISO639-3' in language_info and language_info['ISO639-3']:
            iso_info += f"ISO: {language_info['ISO639-3']}"
        if 'Glottocode' in language_info and language_info['Glottocode']:
            iso_info += f" | Glottolog: {language_info['Glottocode']}"
        
        if iso_info and i == 0:
            main_title += f"\n{iso_info}"
        
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.95)
        
        # Add subtitle with projection info
        proj_name = projection.__class__.__name__.replace('ccrs.', '')
        area_info = f"Center: {center_lat:.2f}°N, {center_lon:.2f}°E | Span: {max_span:.2f}°"
        plt.figtext(0.5, 0.02, f"Projection: {proj_name} | {area_info}", 
                    ha='center', fontsize=8, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the map with group number if multiple groups
        safe_name = "".join(c for c in language_info['Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if len(polygon_groups) > 1:
            filename = f"{safe_name.replace(' ', '_')}_region_{i+1}_cached.png"
        else:
            filename = f"{safe_name.replace(' ', '_')}_cached.png"
        
        filepath = Path(output_dir) / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {filepath}")
        created_maps.append(filepath)
    
    return created_maps

def test_romansh_cached():
    """Test Romansh using cached data (much faster)"""
    print("🇨🇭 Testing Romansh from Cache (Fast Mode)")
    print("=" * 45)
    
    # Try to create maps from cache
    maps = create_maps_from_cache(specific_languages=["Romansh"])
    
    if maps:
        print(f"✅ Successfully created {len(maps)} Romansh maps from cache!")
    else:
        print("❌ No cached data found. Building cache first...")
        # Fall back to regular method and cache the result
        test_romansh()

if __name__ == "__main__":
    print("DiACL Language Region Mapper with Caching")
    print("=" * 50)
    
    # Check if cache exists
    quick_cache_test()
    
    print("\n" + "=" * 50)
    
    # If cache exists, use it for fast counting and map generation

    print("🐌 SLOW MODE: No cache found")
    print("\nOptions:")
    print("1. create_language_cache(max_languages=5)  # Create small cache for testing")
    print("2. create_language_cache()                # Create full cache (takes time)")
    print("3. test_romansh()                         # Just test Romansh (no cache)")
    print("\nBuilding small test cache...")
    
    # Create small cache for demonstration
    create_language_cache(max_languages=1000)
    
    print("\n" + "=" * 30)
    print("✅ Cache created! Run script again for fast mode.")
    
    print("\n🌍 Enhanced features:")
    print("✅ Smart polygon grouping (max 10 regions)")
    print("✅ Globe-like projections with world overview")
    print("✅ Intelligent caching system (JSON)")
    print("✅ Fast map generation from cached data")
    print("✅ Region counting and analysis")
    print("✅ Perfect for creating Anki language cards!")
    
    print("\n💡 Cache Management:")
    print("- create_language_cache(): Build comprehensive cache")
    print("- analyze_cached_data(): Analyze cached statistics") 
    print("- count_total_regions_cached(): Fast region counting")
    print("- create_maps_from_cache(): Generate maps from cache")

# Requirements:
# pip install matplotlib cartopy shapely requests pathlib