import csv
import numpy as np

def calcul_SMD(mean_1, mean_2, sd_1, sd_2):
    numerator = abs(mean_1 - mean_2)
    denominator = np.sqrt((sd_1**2+sd_2**2)/2)
    
    return(numerator/denominator)

short_liste_especes_2009_2018 = ['Capsella bursa-pastoris', 'Conyza', 'Hordeum murinum', 'Plantago major', 'Polygonum aviculare', 'Sisymbrium irio', 'Stellaria media', 'Taraxacum'] 
short_liste_especes_2014_2018 = ['Geranium molle', 'Lolium perenne', 'Poa annua', 'Sonchus oleraceus']

liste_rues = ['BARO', 'BERC', 'CHAR', 'DAUM', 'KESS', 'POMM', 'RAPE', 'RBER', 'REUI', 'TAIN']
liste_ecart_ext = [[] for rue in liste_rues]

for rue in liste_rues:
    for espece in short_liste_especes_2009_2018:
        fichier_csv = open('output_analysis_pa_2009_2018/'+espece+'_summary.csv', 'rt')
        reader = csv.reader(fichier_csv)
        
        liste_ext_rue = []
        liste_std_rue = []
        liste_ext_hors_rue = []
        liste_std_hors_rue = []
        
        for row in reader:
            if row[1] != 'GER':
                if row[0][0:4] == rue:
                    liste_ext_rue.append(float(row[3]))
                    liste_std_rue.append(float(row[4]))
                else:
                    liste_ext_hors_rue.append(float(row[3]))
                    liste_std_hors_rue.append(float(row[4]))
        
        if len(liste_ext_rue) > 1:
            for j in range(len(liste_ext_rue)):
                val = liste_ext_rue[j]
                std = liste_std_rue[j]
                
                smd_intra_rue = sum([calcul_SMD(val,liste_ext_rue[i],std,liste_std_rue[i]) for i in range(len(liste_ext_rue))])/(len(liste_ext_rue)-1)
                smd_inter_rue = sum([calcul_SMD(val,liste_ext_hors_rue[i],std,liste_std_hors_rue[i]) for i in range(len(liste_ext_hors_rue))])/(len(liste_ext_hors_rue))
            
                liste_ecart_ext[liste_rues.index(rue)].append([espece,smd_intra_rue/smd_inter_rue])
                
    for espece in short_liste_especes_2014_2018:
        fichier_csv = open('output_analysis_pa_2014_2018/'+espece+'_summary.csv', 'rt')
        reader = csv.reader(fichier_csv)
        
        liste_ext_rue = []
        liste_std_rue = []
        liste_ext_hors_rue = []
        liste_std_hors_rue = []
        
        for row in reader:
            if row[1] != 'GER':
                if row[0][0:4] == rue:
                    liste_ext_rue.append(float(row[3]))
                    liste_std_rue.append(float(row[4]))
                else:
                    liste_ext_hors_rue.append(float(row[3]))
                    liste_std_hors_rue.append(float(row[4]))
        
        if len(liste_ext_rue) > 1:
            for j in range(len(liste_ext_rue)):
                val = liste_ext_rue[j]
                std = liste_std_rue[j]
                
                smd_intra_rue = sum([calcul_SMD(val,liste_ext_rue[i],std,liste_std_rue[i]) for i in range(len(liste_ext_rue))])/(len(liste_ext_rue)-1)
                smd_inter_rue = sum([calcul_SMD(val,liste_ext_hors_rue[i],std,liste_std_hors_rue[i]) for i in range(len(liste_ext_hors_rue))])/(len(liste_ext_hors_rue))
            
                liste_ecart_ext[liste_rues.index(rue)].append([espece,smd_intra_rue/smd_inter_rue])
                
print(liste_ecart_ext[0])

csv_resultats = open('smd_proba_ext.csv', 'wt')
writer = csv.writer(csv_resultats)

writer.writerow(['street', 'species', 'ecart_relatif_smd_p_ext'])

for i in range(len(liste_rues)):
    for elem in liste_ecart_ext[i]:
        writer.writerow([liste_rues[i]]+elem)

csv_resultats.close()
