import csv

liste_especes_2009_2018 = ['Capsella bursa-pastoris', 'Chenopodium album', 'Conyza', 'Hordeum murinum', 'Lactuca serriola', 'Plantago lanceolata', 'Plantago major', 'Polygonum aviculare', 'Senecio inaequidens', 'Senecio vulgaris', 'Sisymbrium irio', 'Stellaria media', 'Taraxacum', 'Veronica persica'] 
liste_especes_2014_2018 = ['Bromus sterilis', 'Geranium molle', 'Lolium perenne', 'Parietaria judaica', 'Poa annua', 'Sisymbrium officinale', 'Sonchus oleraceus']

liste_extinction_probas = []

for espece in liste_especes_2009_2018:
    csv_file = open('output_analysis_pa_2009_2018/'+espece+'_summary.csv', 'rt')
    reader = csv.reader(csv_file)
    
    for row in reader:
        if row[1] != 'GER':
            liste_extinction_probas.append([espece]+[row[0][0:4]]+row[0:4])
    
    csv_file.close()

for espece in liste_especes_2014_2018:
    csv_file = open('output_analysis_pa_2014_2018/'+espece+'_summary.csv', 'rt')
    reader = csv.reader(csv_file)
    
    for row in reader:
        if row[1] != 'GER':
            liste_extinction_probas.append([espece]+[row[0][0:4]]+row[0:4])
            
    csv_file.close()
    
csv_fusion = open('fusion_csvs.csv', 'wt')
writer = csv.writer(csv_fusion)

writer.writerow(['species', 'street', 'street_complete', 'extinction_risk_1', 'extinction_risk_2', 'p_ext_mean'])

for row in liste_extinction_probas:
    writer.writerow(row)
    
csv_fusion.close()


