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

void reshape(int width, int height)
{
    // set viewport and projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(CAMERA_FOV, (GLdouble)width / height, CAMERA_Z_NEAR, CAMERA_Z_FAR);
}

void display()
{
    // enable depth buffer
    glEnable(GL_DEPTH_TEST);

    // set clear color and depth value
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);

    // clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set random camera view position
    glm::vec4 eye = glm::rotate(360.0f * std::rand() / RAND_MAX, glm::vec3(1.0f, 0.0f, 0.0f)) *
                    glm::rotate(360.0f * std::rand() / RAND_MAX, glm::vec3(0.0f, 1.0f, 0.0f)) *
                    glm::rotate(360.0f * std::rand() / RAND_MAX, glm::vec3(0.0f, 0.0f, 1.0f)) *
                    glm::vec4((float)CAMERA_DIST, 0.0f, 0.0f, 1.0f);

    // set camera view
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye.x, eye.y, eye.z, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    // draw and unproject object to get virtual scan data
    static int i = 0, j = 0;
    draw_obj(dirList.at(i) + "/" + OBJ_PATH);
    unproject(dirList.at(i) + "/" + PCD_PATH);

    // increase scan number
    if (++j >= SCAN_NUM)
    {
        // reset scan number
        j = 0;

        // increase model number
        if (++i >= dirList.size())
        {
            // exit program
            exit(0);    
        }
    }

    // swap buffer
    glutSwapBuffers();
}

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
