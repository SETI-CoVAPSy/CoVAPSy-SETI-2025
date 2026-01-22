#!/bin/bash

# Allow user to provide argument to override default world folder
if [ "$#" -eq 1 ]; then
    WORLD_FOLDER="$1"
    else
    WORLD_FOLDER="Webots_SETI"
fi

export USER="$(whoami)" # Set USER environment variable as current user
webots --stdout --stderr /workspaces/CoVAPSy-SETI-2025/Webots/$WORLD_FOLDER/worlds/Piste_CoVAPSy_2025a.wbt