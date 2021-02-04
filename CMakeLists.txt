cmake_minimum_required(VERSION 3.0.0)
project(shapenet-crn VERSION 0.1.0)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(glm REQUIRED)

include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS})
link_libraries(
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES})

add_executable(vscan vscan/main.cpp)
