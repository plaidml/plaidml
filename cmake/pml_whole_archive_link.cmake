# Copyright 2020 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/iree_whole_archive_link.cmake

# Lists all transitive dependencies of DIRECT_DEPS in TRANSITIVE_DEPS.
function(_pml_transitive_dependencies DIRECT_DEPS TRANSITIVE_DEPS)
  set(_TRANSITIVE "")

  foreach(_DEP ${DIRECT_DEPS})
    _pml_transitive_dependencies_helper(${_DEP} _TRANSITIVE)
  endforeach(_DEP)

  set(${TRANSITIVE_DEPS} "${_TRANSITIVE}" PARENT_SCOPE)
endfunction()

# Recursive helper function for _pml_transitive_dependencies.
# Performs a depth-first search through the dependency graph, appending all
# dependencies of TARGET to the TRANSITIVE_DEPS list.
function(_pml_transitive_dependencies_helper TARGET TRANSITIVE_DEPS)
  if (NOT TARGET "${TARGET}")
    # Excluded from the project, or invalid name? Just ignore.
    return()
  endif()

  # Resolve aliases, canonicalize name formatting.
  get_target_property(_ALIASED_TARGET ${TARGET} ALIASED_TARGET)
  if(_ALIASED_TARGET)
    set(_TARGET_NAME ${_ALIASED_TARGET})
  else()
    string(REPLACE "::" "_" _TARGET_NAME ${TARGET})
  endif()

  set(_RESULT "${${TRANSITIVE_DEPS}}")
  if (${_TARGET_NAME} IN_LIST _RESULT)
    # Already visited, ignore.
    return()
  endif()

  # Append this target to the list. Dependencies of this target will be added
  # (if valid and not already visited) in recursive function calls.
  list(APPEND _RESULT ${_TARGET_NAME})

  # Check for non-target identifiers again after resolving the alias.
  if (NOT TARGET ${_TARGET_NAME})
    return()
  endif()

  # Get the list of direct dependencies for this target.
  get_target_property(_TARGET_TYPE ${_TARGET_NAME} TYPE)
  if(NOT ${_TARGET_TYPE} STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} LINK_LIBRARIES)
  else()
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} INTERFACE_LINK_LIBRARIES)
  endif()

  if(_TARGET_DEPS)
    # Recurse on each dependency.
    foreach(_TARGET_DEP ${_TARGET_DEPS})
      _pml_transitive_dependencies_helper(${_TARGET_DEP} _RESULT)
    endforeach(_TARGET_DEP)
  endif()

  # Propagate the augmented list up to the parent scope.
  set(${TRANSITIVE_DEPS} "${_RESULT}" PARENT_SCOPE)
endfunction()

# Given the ${TARGET} and the libaries it directly depends on in ${ARGN},
# properly establish the linking relationship by considering ALWAYSLINK
# in a recursive manner.
#
# All symbols from ALWAYSLINK libraries will be included in ${TARGET},
# regardless of whether they are directly referenced or not.
function(pml_whole_archive_link TARGET)
  # List all dependencies, including transitive dependencies, then split the
  # dependency list into one for whole archive (ALWAYSLINK) and one for
  # standard linking (which only links in symbols that are directly used).
  _pml_transitive_dependencies("${ARGN}" _TRANSITIVE_DEPS)
  set(_ALWAYS_LINK_DEPS "")
  set(_STANDARD_DEPS "")
  foreach(_DEP ${_TRANSITIVE_DEPS})
    # Check if _DEP is a library with the ALWAYSLINK property set.
    set(_DEP_IS_ALWAYSLINK OFF)
    if (TARGET ${_DEP})
      get_target_property(_DEP_TYPE ${_DEP} TYPE)
      if(${_DEP_TYPE} STREQUAL "INTERFACE_LIBRARY")
        # Can't be ALWAYSLINK since it's an INTERFACE library.
        # We also can't even query for the property, since it isn't allowlisted.
      else()
        get_target_property(_DEP_IS_ALWAYSLINK ${_DEP} ALWAYSLINK)
      endif()
    endif()

    # Append to the corresponding list of deps.
    if(_DEP_IS_ALWAYSLINK)
      list(APPEND _ALWAYS_LINK_DEPS ${_DEP})

      # For MSVC, also add a `-WHOLEARCHIVE:` version of the dep.
      # CMake treats -WHOLEARCHIVE[:lib] as a link flag and will not actually
      # try to link the library in, so we need the flag *and* the dependency.
      # For macOS, also add a `-Wl,-force_load` version of the dep.
      if(MSVC)
        get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
        if (_ALIASED_TARGET)
          list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_ALIASED_TARGET}")
        else()
          list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_DEP}")
        endif()
      elseif(APPLE)
        get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
        if (_ALIASED_TARGET)
          list(APPEND _ALWAYS_LINK_DEPS "-Wl,-force_load $<TARGET_FILE:${_ALIASED_TARGET}>")
        else()
          list(APPEND _ALWAYS_LINK_DEPS "-Wl,-force_load $<TARGET_FILE:${_DEP}>")
        endif()
      endif()
    else()
      list(APPEND _STANDARD_DEPS ${_DEP})
    endif()
  endforeach(_DEP)

  # Call into target_link_libraries with the lists of deps.
  if(MSVC OR APPLE)
    target_link_libraries(${TARGET}
      PUBLIC
        ${_ALWAYS_LINK_DEPS}
        ${_STANDARD_DEPS}
    )
  else()
    target_link_libraries(${TARGET}
      PUBLIC
        "-Wl,--whole-archive"
        ${_ALWAYS_LINK_DEPS}
        "-Wl,--no-whole-archive"
        ${_STANDARD_DEPS}
    )
  endif()
endfunction()
