VERS_1.0 {
    /* Export library symbols. */
    global: ${LINUX_SYMBOLS}

    /* Hide all other symbols. */
    local: *;
};
