from prettytable import PrettyTable 
from operator import attrgetter
import numpy as np
import util

class PrintSorted:
    
    @staticmethod
    def in_table(objs):  
        table = _create_table()
        table = _table_features(table=table)
        _save_file(table=table, objs=objs)
        
def _create_table():
    lista = []
    
    for database in util.DATABASES:
        for metr in util.TABLE.METRICS:
            item = f"{database} - {metr.upper()}(%)"
            lista.append(item)
        
    field_names = util.TABLE.FIELD_NAMES 
    field_names.extend(lista)
    field_names.extend(util.TABLE.MEAN_NAMES)
    
    table = PrettyTable() 
    table.title = util.TABLE.TITLE
    table.field_names = field_names
    
    return table

def _table_features(table):  
    for item in util.TABLE.ITEMS_ALIGN:
        table.align[item] = util.TABLE.CHOSEN_ALIGN
    
    return table

def _save_file(table, objs):
    _add_data_in_table(table=table, objs=objs)

    with open(util.TABLE.PRINT_PATH, 'w') as file:
        file.write(str(table))
                       
def _add_data_in_table(table, objs):
    #organize data
    objs = sorted(objs, key=attrgetter(*util.TABLE.ORGANIZED_DATA))
    
    result = []
    size_metr = len(util.DATABASES)
       
    for j in range(0, len(objs), size_metr):
        name = f"{objs[j].spectrum.upper()}"
        data = [objs[j].model.upper(), name]
        
        metrs_list = []
        f1s_db = []
        ious_db = []
              
        for i in range(0, len(util.DATABASES)):
            f1s = objs[i+j].mean_f1s
            ious = objs[i+j].mean_ious
            
            metrs = [round(f1s*100, 2), round(ious*100, 2)]
            metrs_list.extend(metrs)
            
            f1s_db.append(f1s*100)
            ious_db.append(ious*100)
                 
        mean_f1s_db = round(np.mean(f1s_db), 2)
        mean_ious_db = round(np.mean(ious_db), 2)
        mean_metr_db = [mean_f1s_db, mean_ious_db]
        
        data.extend(metrs_list)
        data.extend(mean_metr_db)
        result.extend(data)
        table.add_row(data)     
    
    _sort_items(table)
    
def _sort_items(table): 
    table.sortby = f"MEAN_{util.TABLE.ITEM_SORT.upper()}(%)"   
    table.reversesort = util.TABLE.SORT