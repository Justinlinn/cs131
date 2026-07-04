#CS131 ws4/.bashrc

alias ll='ls -alh'

mkcd() {
    mkdir -p "$1"
    cd "$1"
}

if [[ $- == *i* ]]; then
    echo "Welcome, $USER"
fi
