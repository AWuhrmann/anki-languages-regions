#!/usr/bin/env python3
"""
Enhanced Map Generator with Distance-Based Grouping
Groups language regions based on geographic distance rather than political boundaries
"""

import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, MultiPolygon
import numpy as np
from pathlib import Path
import time
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

def calculate_centroid_distance_km(poly1, poly2):
    """Calculate distance between polygon centroids in kilometers"""
    centroid1 = poly1.centroid
    centroid2 = poly2.centroid
    
    coord1 = (centroid1.y, centroid1.x)  # (lat, lon)
    coord2 = (centroid2.y, centroid2.x)  # (lat, lon)
    
    return geodesic(coord1, coord2).kilometers

def group_polygons_by_distance(polygons, max_distance_km=1000):
    """
    Group polygons based on distance between their centroids
    
    Args:
        polygons: List of shapely polygons
        max_distance_km: Maximum distance in km to consider polygons as same group
    
    Returns:
        List of polygon groups (each group is a list of polygons)
    """
    if not polygons:
        return []
    
    if len(polygons) == 1:
        return [polygons]
    
    print(f"   🔍 Grouping {len(polygons)} regions (max distance: {max_distance_km}km)")
    
    # Calculate centroids
    centroids = []
    for poly in polygons:
        centroid = poly.centroid
        centroids.append([centroid.y, centroid.x])  # [lat, lon] for DBSCAN
    
    centroids = np.array(centroids)
    
    # Use DBSCAN for clustering based on geographic distance
    # Convert km to approximate degrees (rough conversion)
    # 1 degree ≈ 111 km, but this varies by latitude
    max_distance_degrees = max_distance_km / 111.0
    
    # DBSCAN with geographic distance
    clustering = DBSCAN(
        eps=max_distance_degrees, 
        min_samples=1,  # Each point can form its own cluster
        metric='haversine'  # Great circle distance
    ).fit(np.radians(centroids))  # Convert to radians for haversine
    
    labels = clustering.labels_
    
    # Group polygons by cluster labels
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(polygons[i])
    
    polygon_groups = list(groups.values())
    
    print(f"   📊 Created {len(polygon_groups)} geographic groups")
    
    # Print group information
    for i, group in enumerate(polygon_groups):
        centroids_in_group = [poly.centroid for poly in group]
        if len(centroids_in_group) > 1:
            # Calculate average position
            avg_lat = sum(c.y for c in centroids_in_group) / len(centroids_in_group)
            avg_lon = sum(c.x for c in centroids_in_group) / len(centroids_in_group)
            print(f"     Group {i+1}: {len(group)} regions around ({avg_lat:.2f}, {avg_lon:.2f})")
        else:
            c = centroids_in_group[0]
            print(f"     Group {i+1}: 1 region at ({c.y:.2f}, {c.x:.2f})")
    
    return polygon_groups

def load_language_cache(cache_file="language_cache.json"):
    """Load processed language data from cache file"""
    if not Path(cache_file).exists():
        print(f"❌ Cache file {cache_file} not found")
        print("Run create_language_cache() first!")
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

def calculate_polygon_area_km2(polygon):
    """Calculate approximate area of a polygon in km²"""
    bounds = polygon.bounds
    lat_center = (bounds[1] + bounds[3]) / 2
    
    # Rough conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 * cos(lat) km
    lat_km_per_degree = 111.0
    lon_km_per_degree = 111.0 * abs(np.cos(np.radians(lat_center)))
    
    # Calculate area using shapely (in square degrees) then convert to km²
    area_deg2 = polygon.area
    area_km2 = area_deg2 * lat_km_per_degree * lon_km_per_degree
    
    return area_km2

def create_world_overview_subplot(ax, current_polygon_group, region_index=0):
    """Create a world overview showing only the current region being displayed"""
    
    # Set up world map
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.7, color='lightgray', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.7, color='lightblue', facecolor='lightblue')
    
    # Calculate centroid for the current region group only
    if len(current_polygon_group) == 1:
        centroid = current_polygon_group[0].centroid
    else:
        centroids = [poly.centroid for poly in current_polygon_group]
        avg_lon = sum(c.x for c in centroids) / len(centroids)
        avg_lat = sum(c.y for c in centroids) / len(centroids)
        centroid = Point(avg_lon, avg_lat)
    
    # Plot only the current region with a prominent red dot
    ax.plot(centroid.x, centroid.y, 'o', color='red', markersize=12,
           markeredgecolor='black', markeredgewidth=2,
           transform=ccrs.PlateCarree())

def reconstruct_all_polygons_from_cache(lang_data):
    """Reconstruct ALL polygons from cached WKT data (without pre-existing grouping)"""
    all_polygons = []
    
    # Extract all polygons from all groups
    for group_wkt_list in lang_data['polygon_groups_wkt']:
        for poly_wkt in group_wkt_list:
            polygon = load_wkt(poly_wkt)
            all_polygons.append(polygon)
    
    return all_polygons

def create_maps_from_cached_language_with_distance_grouping(lang_data, output_dir="language_maps", max_distance_km=1000):
    """Create maps for a language using distance-based grouping instead of cached grouping"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    language_name = lang_data['name']
    print(f"🗺️  Creating maps for {language_name} (distance-based grouping)...")
    
    # Reconstruct language info structure
    language_info = {
        'Name': language_name,
        'ISO639-3': lang_data.get('iso639_3', ''),
        'Glottocode': lang_data.get('glottocode', ''),
        'AlternativeNames': lang_data.get('alternative_names', ''),
        'FocalPointWgs84': lang_data.get('focal_point', {})
    }
    
    # Get ALL polygons (ignore existing grouping)
    all_polygons = reconstruct_all_polygons_from_cache(lang_data)
    
    # Apply distance-based grouping
    polygon_groups = group_polygons_by_distance(all_polygons, max_distance_km)
    
    print(f"   Found {len(polygon_groups)} distance-based groups (was {len(lang_data['polygon_groups_wkt'])} cached groups)")
    
    created_maps = []
    
    # Create a map for each polygon group
    for i, polygon_group in enumerate(polygon_groups):
        
        # Combine polygons in this group
        if len(polygon_group) == 1:
            combined_geometry = polygon_group[0]
        else:
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
        
        # Calculate total area of this group
        total_area = sum(calculate_polygon_area_km2(poly) for poly in polygon_group)
        
        print(f"   Group {i+1}: {len(polygon_group)} regions, Center=({center_lat:.2f}, {center_lon:.2f}), Span={max_span:.1f}°, Area={total_area:.0f}km²")
        
        # Choose projection and zoom based on location and size
        if max_span > 60:  # Very large areas
            projection = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
            buffer = max_span * 0.8
            print('maxpsan > 60')
        elif abs(center_lat) > 60:  # Arctic/Antarctic regions
            if center_lat > 0:
                print('center_lat > 60')
                projection = ccrs.NorthPolarStereo(central_longitude=center_lon)
            else:
                print('center_lat < -60')
                projection = ccrs.SouthPolarStereo(central_longitude=center_lon)
            buffer = max(max_span * 1.5, 5)
        else:  # Most regions
            projection = ccrs.AzimuthalEquidistant(central_longitude=center_lon, central_latitude=center_lat)
            
            # Adjust buffer based on area size
            if max_span < 1.0:  # Very small area
                print('maxpsan < 1.0')
                buffer = 20
            elif max_span < 5.0:  # Small area
                print('maxpsan < 5.0')
                buffer = 30
            else:  # Regular area
                print('maxpsan >= 5.0')
                buffer = max(maxpsan * 3, 40)
        
        # Create figure with subplots: main map + world overview
        fig = plt.figure(figsize=(16, 10))
        
        # Main detailed map
        ax_main = plt.subplot(1, 2, 1, projection=projection)
        
        # World overview map
        ax_world = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        
        # === MAIN MAP ===
        if isinstance(projection, ccrs.Orthographic):
            ax_main.set_global()
        else:
            try:
                ax_main.set_extent([
                    center_lon - buffer, center_lon + buffer,
                    center_lat - buffer, center_lat + buffer
                ], ccrs.PlateCarree())
            except:
                ax_main.set_extent([
                    max(-180, center_lon - buffer), min(180, center_lon + buffer),
                    max(-90, center_lat - buffer), min(90, center_lat + buffer)
                ], ccrs.PlateCarree())
        
        # Add map features
        ax_main.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax_main.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6, color='gray')
        ax_main.add_feature(cfeature.LAND, alpha=0.9, color='lightgray', facecolor='lightgray')
        ax_main.add_feature(cfeature.OCEAN, alpha=0.9, color='lightsteelblue', facecolor='lightsteelblue')
        ax_main.add_feature(cfeature.RIVERS, alpha=0.5, linewidth=0.5, edgecolor='blue')
        ax_main.add_feature(cfeature.LAKES, alpha=0.7, color='lightblue', facecolor='lightblue')
        
        # Plot all polygons in this group
        for poly in polygon_group:
            if poly.exterior:
                x_coords, y_coords = poly.exterior.xy
                ax_main.fill(x_coords, y_coords, 
                           color='red', alpha=0.6, 
                           edgecolor='darkred', linewidth=3,
                           transform=ccrs.PlateCarree())
                
                ax_main.plot(x_coords, y_coords,
                           color='darkred', linewidth=4,
                           transform=ccrs.PlateCarree())
        
        # Add grid lines (subtle, no labels for clean appearance)
        if not isinstance(projection, ccrs.Orthographic):
            ax_main.gridlines(draw_labels=False, dms=False, 
                            linewidth=0.3, alpha=0.3, color='gray', linestyle='--')
        
        # === WORLD OVERVIEW ===
        create_world_overview_subplot(ax_world, polygon_group, i)
        
        # Adjust layout without titles
        plt.tight_layout()
        
        # Save the map
        safe_name = "".join(c for c in language_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        if len(polygon_groups) > 1:
            filename = f"{safe_name}_Group_{i+1}.png"
        else:
            filename = f"{safe_name}.png"
        
        filepath = Path(output_dir) / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {filepath}")
        created_maps.append(filepath)
    
    return created_maps

def generate_maps_with_distance_grouping(cache_file="language_cache.json", 
                                       output_dir="language_maps_distance",
                                       max_distance_km=1000,
                                       specific_languages=None,
                                       max_languages=None):
    """Generate all maps using distance-based grouping"""
    
    print(f"🗺️  Map Generator with Distance-Based Grouping")
    print(f"📏 Maximum distance for grouping: {max_distance_km} km")
    print("=" * 50)
    
    # Load cache
    cache_data = load_language_cache(cache_file)
    if not cache_data:
        return []
    
    languages = cache_data['languages']
    
    # Apply filters
    if specific_languages:
        if isinstance(specific_languages, str):
            specific_languages = [specific_languages]
        
        filtered_languages = {}
        for presence_id, lang_data in languages.items():
            if (lang_data['name'] in specific_languages or 
                presence_id in specific_languages):
                filtered_languages[presence_id] = lang_data
        languages = filtered_languages
        
        if not languages:
            print(f"❌ No languages found matching: {specific_languages}")
            return []
    
    if max_languages:
        languages = dict(list(languages.items())[:max_languages])
        print(f"📊 Limited to first {max_languages} languages")
    
    print(f"🎯 Processing {len(languages)} languages...")
    
    all_created_maps = []
    successful_languages = 0
    
    for i, (presence_id, lang_data) in enumerate(languages.items()):
        print(f"\n[{i+1}/{len(languages)}] {lang_data['name']}")
        
        try:
            created_maps = create_maps_from_cached_language_with_distance_grouping(
                lang_data, output_dir, max_distance_km)
            all_created_maps.extend(created_maps)
            successful_languages += 1
            
        except Exception as e:
            print(f"   ❌ Error creating maps: {str(e)[:50]}...")
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    print(f"\n🎉 COMPLETE!")
    print(f"📊 Results:")
    print(f"   Languages processed: {successful_languages}/{len(languages)}")
    print(f"   Total maps created: {len(all_created_maps)}")
    print(f"   Maps saved in: {output_dir}/")
    print(f"📏 Distance threshold used: {max_distance_km} km")
    
    return all_created_maps

def test_distance_grouping_examples():
    """Test distance grouping with some example languages"""
    
    # Test with different distance thresholds
    test_distances = [500, 800, 1000, 1500]
    test_languages = ["French", "English", "Portuguese", "Spanish"]
    
    for distance in test_distances:
        print(f"\n🧪 Testing with {distance}km threshold...")
        generate_maps_with_distance_grouping(
            max_distance_km=distance,
            specific_languages=test_languages,
            output_dir=f"test_distance_{distance}km"
        )

if __name__ == "__main__":
    print("Enhanced Map Generator with Distance-Based Grouping")
    print("=" * 52)
    
    # Example: Test with French to see the grouping effect
    print("🇫🇷 Testing with French (should separate Europe, Africa, Caribbean, etc.)")
    generate_maps_with_distance_grouping(
        specific_languages=["Aguaruna", "Adyghe", "Yaminahua"],
        max_distance_km=1000,  # 1000km threshold
        output_dir="french_distance_test"
    )
    
    print("\n🎯 Available functions:")
    print("- generate_maps_with_distance_grouping(max_distance_km=1000)  # All languages with distance grouping")
    print("- test_distance_grouping_examples()                          # Test different thresholds")
    print("- generate_maps_with_distance_grouping(specific_languages=['French'], max_distance_km=800)")
    
    print(f"\n📏 Recommended distance thresholds:")
    print(f"   500km  - Very strict grouping (separate most islands)")
    print(f"   800km  - Moderate grouping (separate continents)")  
    print(f"   1000km - Balanced grouping (recommended)")
    print(f"   1500km - Loose grouping (keep some distant territories together)")