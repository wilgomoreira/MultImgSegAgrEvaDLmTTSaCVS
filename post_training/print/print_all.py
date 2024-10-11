from print.print_sorted import PrintSorted
from print.print_performance import PrintPerformance
from print.print_ece_metr import PrintEceMetr

class PrintAll:
    @staticmethod
    def files(objs):
        PrintSorted.in_table(objs=objs)
        PrintPerformance.in_sheet(objs=objs)
       # PrintEceMetr.in_sheet(objs=objs)
        
        



  
    
