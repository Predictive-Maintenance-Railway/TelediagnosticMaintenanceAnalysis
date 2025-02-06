# packages/frameworks:
#import pandas as pd
#import geopandas as gpd
#import folium
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
#from datetime import datetime, timedelta

'''
Firstly, drop missing values in lat and lot:
# 1) drop missing values for lat and lot:
print("Dataset:")
print(f"tot n. of obs: {len(data)}")
print(f"n. of lat=0: {(data['lat'] == 0).sum()}")
print(f"n. of lon=0: {(data['lon'] == 0).sum()}")
print(f"n. of lat=0 and lon=0: {((data['lat'] == 0) & (data['lon'] == 0)).sum()}")
lat_lon = data[data['lat'] != 0]
lat_lon2 = lat_lon[lat_lon['lon'] != 0]
lat_lon3 = lat_lon2[(lat_lon2['lat'] != 0) & (lat_lon2['lon'] != 0)]
lat_lon3.reset_index(drop=True, inplace=True)
print("\n")
print("Dataset without missing lat and lot:")
print(f"tot n. of obs: {len(lat_lon3)}")
print(f"n. of lat=0: {(lat_lon3['lat'] == 0).sum()}")
print(f"n. of lon=0: {(lat_lon3['lon'] == 0).sum()}")
print(f"n. of lat=0 and lon=0: {((lat_lon3['lat'] == 0) & (lat_lon3['lon'] == 0)).sum()}")
print("\n")

Then, understand the measurement unit:
# 3) measure unit?
print(f"Trenord coordinates:")
print(f"first 10 unique values lat: {sorted(data['lat'].unique()[:10])}")
print(f"first 10 unique values lon: {sorted(data['lon'].unique()[:10])}")
print("= DECIMAL DEGREE")
print("\n")
print(f"ISTAT coordinates:")
print(lombardia["geometry"])
print("= Universal Transverse Mercator (UTM)")
'''


# A) Italy plot + scatterplot:

# 1) Load administrative boundaries from ISTAT:
Regioni = gpd.read_file("..//Confini_amministrativi_ISTAT/Regioni/Reg01012024_WGS84.shp")
#print(f"Regioni: \n {Regioni.head()}")
#print("\n")
#print(f"Nomi regioni: \n {Regioni['DEN_REG'].unique()}")
#print("\n")

Province = gpd.read_file("..//Confini_amministrativi_ISTAT/Province/ProvCM01012024_WGS84.shp")
#print(f"Province: \n {Province.head()}")
#print("\n")

Comuni = gpd.read_file("..//Confini_amministrativi_ISTAT/Comuni/Com01012024_WGS84.shp")
#print(f"Comuni: \n {Comuni.head()}")
#print("\n")

focus = ["Varese", "Como", "Monza", "Monza e della Brianza", "Milano", "Lodi", "Pavia", "Novara"]

Lombardia = Regioni[Regioni['DEN_REG'] == "Lombardia"]
Province_lombardia = Province[Province['COD_REG'] == 3]
Province_lombardia_to_plot = Province_lombardia[Province_lombardia['DEN_UTS'].isin(focus)]
Comuni_lombardia = Comuni[Comuni['COD_REG'] == 3]
Comuni_lombardia_to_plot = Comuni_lombardia[Comuni_lombardia['COMUNE'].isin(focus)]


Pimonte = Regioni[Regioni['DEN_REG'] == "Piemonte"]
Province_piemonte = Province[Province['COD_REG'] == 1]
Province_piemonte_to_plot = Province_piemonte[Province_piemonte['DEN_UTS'].isin(focus)]
Comuni_piemonte = Comuni[Comuni['COD_REG'] == 1]
Comuni_piemonte_to_plot = Comuni_piemonte[Comuni_piemonte['COMUNE'].isin(focus)]

print(f"Focus province: \n {Province_lombardia_to_plot} {Province_piemonte_to_plot}")
print("\n")
print(f"focus comuni: \n {Comuni_lombardia_to_plot} {Comuni_piemonte_to_plot}")


# 2) Plot Italy, Lombardia, provinces and municipalities, GPS coordinates of each alert occured:
# PLot:
fig, ax = plt.subplots(figsize=(10, 10))
Regioni.plot(ax=ax, color='lightgray', edgecolor='black')                   				# Italy with regions boundaries
Province_lombardia_to_plot.plot(ax=ax, color='none', edgecolor='green')     				# highlighting provinces
Comuni_lombardia_to_plot.plot(ax=ax, color='none', edgecolor=['blue', 'dodgerblue', 'magenta', 'darkorange', 'peru', 'rebeccapurple'])      # highlights districts
Province_piemonte_to_plot.plot(ax=ax, color='none', edgecolor='green')
Comuni_piemonte_to_plot.plot(ax=ax, color='none', edgecolor='pink')
Lombardia.plot(ax=ax, color='none', edgecolor='red', linewidth=0.5)							# highlighting Lombardia
plt.scatter(data.Lat_umt, data.Lon_umt, color='yellow', marker='o', s=8, alpha=0.3)   				# alerts' coordinates
ax.set_xlim(0.4e6, 0.8e6)
ax.set_ylim(4.8e6, 5.2e6)
custom_lines = [Line2D([0], [0], color='pink', linestyle='-'),
				Line2D([0], [0], color='blue', linestyle='-'),
				Line2D([0], [0], color='dodgerblue', linestyle='-'),
				Line2D([0], [0], color='magenta', linestyle='-'),
				Line2D([0], [0], color='darkorange', linestyle='-'),
				Line2D([0], [0], color='peru', linestyle='-'),
				Line2D([0], [0], color='rebeccapurple', linestyle='-'),
				]
plt.legend(custom_lines, ['Novara', 'Pavia', 'Milano', 'Varese', 'Como', 'Lodi', 'Monza'], facecolor='white', title='Districts',
		   fontsize='medium', title_fontsize='large');


'''
# B) Interactive map:
def draw_map_with_points(df, latitude_column, longitude_column):
     """
     Disegna una mappa con punti utilizzando le coordinate di longitudine e latitudine da un DataFrame pandas.

     :param df: DataFrame pandas contenente le coordinate
     :param latitude_column: Nome della colonna nel DataFrame che contiene le latitudini
     :param longitude_column: Nome della colonna nel DataFrame che contiene le longitudini
     """
     # Prendi il centro della mappa dalle coordinate medie
     center_lat = df[latitude_column].mean()
     center_lon = df[longitude_column].mean()

     # Crea la mappa centrata
     mappa = folium.Map(location=[center_lat, center_lon], zoom_start=6)

     # Aggiungi punti alla mappa
     for _, row in df.iterrows():
         folium.Marker(location=[row[latitude_column],
row[longitude_column]]).add_to(mappa)

     return mappa

data = pd.read_csv('mappa.csv', delimiter=',').tail(3000)# head(3000)

value =  0.143897
p = data.loc[data['lon'] == value]
print(p.index)
senzafrancia = data.drop(p.index)

unique_gps = senzafrancia[['lat', 'lon']].drop_duplicates() mappa = draw_map_with_points(unique_gps, 'lat', 'lon')
mappa.save("mappaOpen.html")
'''