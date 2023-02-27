#!/bin/bash

#
# Wrapper script to setup a target virtualenv and run a python script.
#

# Define usage
function usage
{
  echo -e "\e[1mUsage:\e[21m"
  echo -e "    python.sh [-h] --script PYSCRIPT"
  echo -e "              [--venv VIRTUALENV]"
  echo -e ""
  echo -e "\e[1mDescription:\e[21m"
  echo -e "    TODO."
  echo -e ""
  echo -e "\e[1mArguments:\e[21m"
  echo -e "  -h, --help            show this help message and exit"
  echo -e "  --script PYSCRIPT"
  echo -e "        Python script to be executed."
  echo -e "  --venv VIRTUALENV"
  echo -e "        Run in a specific virtualenv environment."
  echo -e ""
  echo -e "\e[1mExample:\e[21m"
  echo -e "./python.sh \ "
  echo -e "--script graph.py \ "
  echo -e "--venv noesis"
}

# Parse arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case "$key" in
    --help)
      usage
      exit
    ;;
    --script)
      PYSCRIPT="$2"
      shift # past argument
      shift # past value
    ;;
    --venv)
      VIRTUALENV=$2
      shift # past argument
      shift # past value
    ;;
    *) # unknown option
    echo -e "\e[1m\e[31mERROR\e[97m:\e[0m Invalid argument.\e[33m"
    usage
    exit
    ;;
  esac
done

# Argument checks
if ! [[ -v PYSCRIPT ]];
then
 echo -e "\e[1m\e[31mERROR\e[97m:\e[0m Missing argument: --script not specified.\e[33m"
 usage
 exit
fi

# Argument checks
if ! [[ -v VIRTUALENV ]];
then
 echo -e "Using default virtualenv: 'noesis'"
 VIRTUALENV=noesis
else
   echo -e "Using virtualenv: ${VIRTUALENV}"
fi

# Activate the Virtualenv python3.5 environment
source $WORKON_HOME/${VIRTUALENV}/bin/activate

# Run the generation script using the collected arguments
python ${PYSCRIPT}

# EOF
