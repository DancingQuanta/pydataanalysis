import os
import shutil
from subprocess import check_output, CalledProcessError, STDOUT
# import papermill as pm
from src.utils import mkdir


def pandoc_filters():
    # Using a list of filters to select filters from user config directory
    filters = [
        'pandoc-svg.py'
    ]
    
    # User config directory
    cfg_path = os.path.join(os.path.expanduser("~"), ".pandoc", "filters") 
    
    # build pandoc arguments
    args = [f"--filter {os.path.join(cfg_path, filt)}" for filt in filters]
    
    # Convert args to strings separated by a space
    args = ' '.join(args)
    return args
    
def exec_cmd(cmd, debug=False):
    if debug:
        print("Command: {}\n".format(cmd))
    success = False
    try:
        output = check_output(
            cmd, stderr=STDOUT, shell=True,
            universal_newlines=True)
    except CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
    else:
        if debug:
            print("Output: \n{}\n".format(output))
        success = True

    return success

def export_tex(src, dest, debug=False):
    out_dir = os.path.dirname(dest)

    params = {
        'nb_file': src,
        'tex_file': dest + '.tex',
        'doc_file': dest + '.docx',
        'output_directory': out_dir,
    }

    args = [
        'jupyter',
        'nbconvert',
        '--no-prompt',
        '--SVG2PDFPreprocessor.enabled=False',
        '--TagRemovePreprocessor.enabled=True',
        '--TagRemovePreprocessor.remove_input_tags="{{\'remove_input\'}}"',
        '--TagRemovePreprocessor.remove_cell_tags="{{\'remove_cell\', \'injected-parameters\'}}"',
        '--to=latex',
        '--output="{tex_file}"',
        '"{nb_file}"'
    ]
    cmd = ' '.join(args).format(**params)

    success = exec_cmd(cmd, debug)

#     if os.path.isfile(params['tex_file']) and success:
#         args = [
#             'pdflatex.exe',
#             '-interaction=nonstopmode',
#             '-output-directory',
#             '"{output_directory}"',
#             '"{tex_file}"'
#         ]
#         cmd = " ".join(args).format(**params)
#         cmd = cmd.replace('\\', '/')
#         exec_cmd(cmd, debug)

    if os.path.isfile(params['tex_file']) and success:
        args = [
            'pandoc.exe',
            pandoc_filters(),
            '-o "{doc_file}"',
            '"{tex_file}"'
        ]
        cmd = " ".join(args).format(**params)
        cmd = cmd.replace('\\', '/')
        exec_cmd(cmd, debug)

def export_html(src, dest, debug=False):
    out_dir = os.path.dirname(dest)

    params = {
        'nb_file': src,
        'html_file': dest + '.html',
        'doc_file': dest + '.docx',
        'output_directory': out_dir,
    }

    args = [
        'jupyter',
        'nbconvert',
        '--log-level="DEBUG"',
        '--no-prompt',
        '--TagRemovePreprocessor.enabled=True',
        '--TagRemovePreprocessor.remove_input_tags="{{\'remove_input\'}}"',
        '--TagRemovePreprocessor.remove_cell_tags="{{\'remove_cell\', \'injected-parameters\'}}"',
        '--to=html',
        '--output="{html_file}"',
        '"{nb_file}"'
    ]
    cmd = ' '.join(args).format(**params)

    success = exec_cmd(cmd, debug)

    if os.path.isfile(params['html_file']) and success:
        args = [
            'pandoc.exe',
            '-o "{doc_file}"',
            '"{html_file}"'
        ]
        cmd = " ".join(args).format(**params)
        cmd = cmd.replace('\\', '/')
        exec_cmd(cmd, debug)

def export_doc(src, dest, debug=False):
    out_dir = os.path.dirname(dest)

    params = {
        'nb_file': src,
        'md_file': dest + '.md',
        'html_file': dest + '.html',
        'doc_file': dest + '.docx',
        'output_directory': out_dir,
    }

    args = [
        'jupyter',
        'nbconvert',
        '--log-level="DEBUG"',
        '--no-prompt',
        '--TagRemovePreprocessor.enabled=True',
        '--TagRemovePreprocessor.remove_input_tags="{{\'remove_input\'}}"',
        '--TagRemovePreprocessor.remove_cell_tags="{{\'remove_cell\', \'injected-parameters\'}}"',
        '--to=markdown',
        '--output="{md_file}"',
        '"{nb_file}"'
    ]
    cmd = ' '.join(args).format(**params)

    success = exec_cmd(cmd, debug)

#     if os.path.isfile(params['html_file']) and success:
#         args = [
#             'pandoc.exe',
#             '-s',
#             '--mathjax',
#             '-o "{html_file}"',
#             '"{md_file}"'
#         ]
#         cmd = " ".join(args).format(**params)
#         cmd = cmd.replace('\\', '/')
#         exec_cmd(cmd, debug)
        
    if os.path.isfile(params['md_file']) and success:
        args = [
            'pandoc.exe',
            '-o "{doc_file}"',
            '"{md_file}"'
        ]
        cmd = " ".join(args).format(**params)
        cmd = cmd.replace('\\', '/')
        exec_cmd(cmd, debug)

def copy_file(source, target):

    # adding exception handling
    try:
        shutil.copy(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
        
def run_notebook(nb_dest, nb_template, parameters, debug=False):
    
#     # Run notebook from a template
#     pm.execute_notebook(
#         nb_template,
#         nb_dest,
#         parameters=parameters
#     )
    
    # Copy file instead of papermill
    copy_file(nb_template, nb_dest)
    
    src = os.path.normpath(nb_dest)
    d = os.path.dirname(src)
    f = os.path.basename(src)
    fn = os.path.splitext(f)[0]
    out_dir = mkdir(d, fn)
    dest = os.path.join(out_dir, fn)
    
    export_tex(src, dest, debug)
#     export_doc(src, dest, debug)
    
#     export_tex(nb_dest, debug)
#     export_html(nb_dest, debug)

    
