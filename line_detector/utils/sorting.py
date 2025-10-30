import os
import operator


def sort_detection_label(txt_loc, sort_label, flag):
  txt_files = os.listdir(txt_loc)
  obj = Line_sort(txt_files, txt_loc, sort_label, flag)
  
class Line_sort:
    def __init__(self, txt_files, txt_loc, sort_label, flag):
        self.txt_files = txt_files
        self.txt_loc = txt_loc
        self.sort_label = sort_label
        self.flag = flag
        self.read_file()

    def read_file(self):
        files = self.txt_files
        # os.mkdir('/content/sorted_line_after_1st_detection')
        os.mkdir(self.sort_label)
        for file in files:
            txt_file = []
            file_loc = self.txt_loc+file
            with open(file_loc, 'r' , encoding='utf-8',errors='ignore') as lines:
                for line in lines:
                    token = line.split()
                    
                    _, x, y, w, h, conf = map(float, line.split(' '))
                    # print("width -> ",w)
                    # print("confidence -> ",conf)
                    if self.flag == 0: # 1st line detection lavel
                      if w > 0.50 and conf < 0.50:
                        continue
                      else:
                        txt_file.append(token)
                    else: # Word detection lavel
                      # if w > 0.50:
                      #   continue
                      # else:
                        txt_file.append(token)

            if self.flag == 0: # 1st line detection lavel
               sorted_txt_file = sorted(txt_file, key=operator.itemgetter(2))
            else: # Word detection lavel
               sorted_txt_file = sorted(txt_file, key=operator.itemgetter(1))

            # lenght = len(sorted_txt_file[0])
            self.file_write(sorted_txt_file, file)

    def file_write(self,txt_file, file_name):
        # loc = '/content/sorted_line_after_1st_detection/'+file_name
        loc = self.sort_label+file_name
        with open(loc, 'w') as f: 
            c=0
            for line in txt_file:  
                for l in line:
                    c+=1
                    if c == len(line):
                        f.write('%s' % l)
                    else:
                        f.write('%s ' % l)
                f.write("\n")
                c=0

def line_sort(lines):
    sort_lines = {}
    for line in lines:
        img_lb = line.split('.')[0]
        lb = [int(i) for i in img_lb.split('_')]
        new_lb = [ '0'+str(r) if r<10 else str(r) for r in lb]
        if len(new_lb)==3:
            items = int(new_lb[0]+new_lb[1]+new_lb[2])
        if len(new_lb)==4:
            items = int(new_lb[0]+new_lb[1]+new_lb[2]+new_lb[3])
        sort_lines[items] = line
    # print(sort_lines)
    sort_lines = dict(sorted(sort_lines.items()))
    new_lines = list(sort_lines.values())
    return new_lines