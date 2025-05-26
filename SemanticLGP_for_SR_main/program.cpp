#include <Python.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "program.h"
#include <pybind11/embed.h>


// Fonction d'appel du script python avec arguments et récupération du float imprimé
double call_python_script_with_args(const std::vector<std::string>& args) {
    // Initialiser Python (si pas déjà fait)
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // Construire la commande Python pour simuler un appel via sys.argv
    std::ostringstream oss;
    oss << "import sys\n";

    // Simuler sys.argv = ["subprocess_for_genetic.py", arg1, arg2, ...]
    oss << "sys.argv = [\"subprocess_for_genetic.py\"";
    for (const auto& arg : args) {
        oss << ", \"" << arg << "\"";
    }
    oss << "]\n";

    // Rediriger la sortie stdout pour capturer print(mean_val)
    oss << "import io\n"
           "import contextlib\n"
           "f = io.StringIO()\n"
           "with contextlib.redirect_stdout(f):\n"
           "    import subprocess_for_genetic\n"  // Remplacer par le nom réel du fichier sans .py
           "\n"
           "output = f.getvalue()\n";

    // Expose la sortie comme variable Python
    oss << "result_str = output.strip()\n";

    // Exécuter la chaîne python
    std::string py_code = oss.str();
    int ret = PyRun_SimpleString(py_code.c_str());
    if (ret != 0) {
        std::cerr << "Erreur exécution script Python\n";
        return -1.0;
    }

    // Récupérer la variable result_str
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* main_dict = PyModule_GetDict(main_module);
    PyObject* py_result = PyDict_GetItemString(main_dict, "result_str");
    if (!py_result) {
        std::cerr << "Erreur récupération résultat Python\n";
        return -1.0;
    }
    const char* result_cstr = PyUnicode_AsUTF8(py_result);
    if (!result_cstr) {
        std::cerr << "Erreur conversion résultat Python\n";
        return -1.0;
    }

    // Convertir le string en double
    double val = atof(result_cstr);

    // Ne pas finaliser Python ici si tu veux réutiliser plusieurs fois

    return val;
}

double call_python_script_with_args(const std::vector<std::string>& args);
