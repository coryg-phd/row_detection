import geopandas as gpd
import collections
import numpy as np
import math
# import warnings
# warnings.filterwarnings("ignore")



ct_rows = gpd.read_file('./12_CisternNorth_rows_4326_1m_buffer.shp')
ct_obj = gpd.read_file('./12_CisternNorth_results_manual_clean.shp')
ct_obj['row_counts'] = None

aggs = collections.defaultdict(list)
species = ct_obj['Class'].unique()
species = np.append(species, 'id')

for x in species:
  aggs[x].append('size')

for x in species[:-1]:
  ct_obj[x] = None

inter_df = gpd.sjoin(ct_rows, ct_obj, how='left', op='intersects')

count_inter = gpd.GeoDataFrame(inter_df.groupby(['id', 'Class', inter_df['geometry'].to_wkt()]))
print(count_inter)

# for k,v in aggs.items():
#   count_inter = inter_df.groupby([k, inter_df['geometry'].to_wkt()])
#   counts = 


# count_inter = gpd.GeoDataFrame(inter_df.groupby(['id', inter_df['geometry'].to_wkt()]))
# counts = count_inter.size().to_frame(name='counts')
# for i, (k,v) in enumerate(aggs.items()):
#   print(k)
#   counts.join(count_inter.agg({k:v}).rename(columns={k:f'{k}_counts'})).reset_index()
#   print(counts)

# print(count_inter)
# count_inter.columns = ['id', 'geometry', 'row_counts']
# count_inter['geometry'] = gpd.GeoSeries.from_wkt(count_inter['geometry'])
# count_inter = gpd.GeoDataFrame(count_inter)
# count_inter.to_file('./12_CisternNorth_rowCounts.shp', crs=4326, driver='ESRI Shapefile')

