#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define WINDOW_NAME     "Virtual Scan"
#define WINDOW_WIDTH    640
#define WINDOW_HEIGHT   480

#define CAMERA_FOV      45.0
#define CAMERA_Z_NEAR   0.1
#define CAMERA_Z_FAR    5.0
#define CAMERA_DIST     2.5

#define PCD_SIZE        2048
#define SCAN_NUM        20

#define LIST_PATH       "../vscan/list.txt"
#define OBJ_PATH        "models/model_normalized_tri.obj"
#define PCD_PATH        ("pcds/pcd_vscan_" + std::to_string(j) + ".txt")

int main(int argc, char **argv)
{
    // read list file
    read_list(LIST_PATH);

    // initialize freeglut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    // create window
    glutCreateWindow(WINDOW_NAME);

    // set callback functions
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutIdleFunc(display);

    // start main loop
    glutMainLoop();

    // exit program
    return 1;
}
