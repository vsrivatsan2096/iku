import imp
import subprocess

try:
    imp.find_module("jupyterthemes")
except:
    subprocess.check_call(["python", '-m', 'pip', 'install', 'jupyterthemes']) # install pkg
    subprocess.check_call(["python", '-m', 'pip', 'install',"--upgrade", 'jupyterthemes']) # upgrade pkg

available_themes = [ "chesterish", "grade3", "gruvboxd", "gruvboxl", "monokai", "oceans16", "onedork", "solarizedd", "solarizedl"]
for i, theme in enumerate(available_themes):
    print(str(i)+".", theme)
n = int(input("Enter the theme number : "))
subprocess.check_call(["jt", "-t" ,available_themes[n]])
print("Applied "+available_themes[n])
