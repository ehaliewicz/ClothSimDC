#include <dc/fmath.h> /* Fast math library headers for optimized mathematical functions */
#include <dc/pvr.h> /* PVR library headers for PowerVR graphics chip functions */
#include <kos.h> /* Includes necessary KallistiOS (KOS) headers for Dreamcast development */
#include <png/png.h> /* PNG library headers for handling PNG images */
#include <stdio.h> /* Standard I/O library headers for input and output functions */
#include <stdlib.h> /* Standard library headers for general-purpose functions, including abs() */

#include <dc/matrix.h> /* Matrix library headers for handling matrix operations */
#include <dc/matrix3d.h> /* Matrix3D library headers for handling 3D matrix operations */

#include "perspective.h" /* Perspective projection matrix functions */
#include "pvrtex.h"
#include "cube.h"

#define ABS(x) ((x) < 0 ? -(x) : (x))

#define MODEL_SCALE 5.0f
#define DEFAULT_FOV 45.0f
#define MIN_ZOOM -10.0f
#define MAX_ZOOM 15.0f
#define ZOOM_SPEED 0.3f


extern uint8 romdisk[];
KOS_INIT_FLAGS(INIT_DEFAULT | INIT_MALLOCSTATS);
KOS_INIT_ROMDISK(romdisk);

static float fovy = DEFAULT_FOV;

static dttex_info_t texture;

static inline void init_cloth_context(pvr_poly_cxt_t *cxt) {
  pvr_poly_cxt_txr(cxt, PVR_LIST_TR_POLY, texture.pvrformat, texture.width,
                   texture.height, texture.ptr, PVR_FILTER_BILINEAR);
  //pvr_poly_cxt_col(cxt, PVR_LIST_OP_POLY);
  cxt->gen.culling = PVR_CULLING_NONE; // disable culling for polygons facing
                                       // away from the camera
  cxt->gen.shading = PVR_SHADE_GOURAUD;
}

#define CLOTH_NUM_VERTS_ONE_AXIS 64
#define CLOTH_NUM_POLYS_ONE_AXIS (CLOTH_NUM_VERTS_ONE_AXIS-1)
#define CLOTH_SPACING (1.08f * (2.0f / CLOTH_NUM_POLYS_ONE_AXIS))
#define CLOTH_TOTAL_NUM_VERTS (CLOTH_NUM_VERTS_ONE_AXIS * CLOTH_NUM_VERTS_ONE_AXIS)
//vec3f_t cloth_verts[CLOTH_TOTAL_NUM_VERTS];

vec3f_t gravity = { .x = 0.0f, .y = .00035f, .z = 0.0f };


vec3f_t cloth_verts1[CLOTH_TOTAL_NUM_VERTS] __attribute__((aligned(32)));
vec3f_t cloth_verts2[CLOTH_TOTAL_NUM_VERTS] __attribute__((aligned(32)));
vec3f_t *cur_cloth_verts;
vec3f_t *next_cloth_verts;
vec3f_t cloth_prev_verts[CLOTH_TOTAL_NUM_VERTS] __attribute__((aligned(32)));
uint8_t point_is_locked[CLOTH_TOTAL_NUM_VERTS];

vec3f_t cloth_normals[CLOTH_TOTAL_NUM_VERTS] __attribute__((aligned(32)));
uint32 cloth_vert_argbs[CLOTH_TOTAL_NUM_VERTS];

typedef struct {
  uint16 fst, snd;
} segment;
segment segments[(CLOTH_NUM_VERTS_ONE_AXIS * (CLOTH_NUM_VERTS_ONE_AXIS-1)) * 2];



void submit_quad(pvr_dr_state_t dr_state,
                 float x0, float y0, float z0,
                 float x1, float y1, float z1,
                 float x2, float y2, float z2,
                 float x3, float y3, float z3,
                 float u0, float u1, float v0, float v1,
                 uint32_t argb0, uint32_t oargb0, 
                 uint32_t argb1, uint32_t oargb1, 
                 uint32_t argb2, uint32_t oargb2, 
                 uint32_t argb3, uint32_t oargb3) {
                  
    //printf("%f %f %f\n", x0, y0, z0);
    //printf("%f %f %f\n", v_bl.x, v_bl.y, v_bl.z);
    //printf("%f %f %f\n", v_tr.x, v_tr.y, v_tr.z);
    //printf("%f %f %f\n", v_br.x, v_br.y, v_br.z);
                  
    pvr_vertex_t *vert;
  
    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX;
    vert->x = x0;
    vert->y = y0;
    vert->z = z0;
    vert->u = u0;
    vert->v = v0;
    vert->argb = argb0;
    vert->oargb = oargb0;
    pvr_dr_commit(vert);
    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX;
    vert->x = x1;
    vert->y = y1;
    vert->z = z1;
    vert->u = u0;
    vert->v = v1;
    vert->argb = argb1;
    vert->oargb = oargb1;
    pvr_dr_commit(vert);
    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX;
    vert->x = x2;
    vert->y = y2;
    vert->z = z2;
    vert->u = u1;
    vert->v = v0;
    vert->argb = argb2;
    vert->oargb = oargb2;
    pvr_dr_commit(vert);
    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX_EOL;
    vert->x = x3;
    vert->y = y3;
    vert->z = z3;
    vert->u = u1;
    vert->v = v1;
    vert->argb = argb3;
    vert->oargb = oargb3;
    pvr_dr_commit(vert);
}

vec3f_t trans_verts[CLOTH_TOTAL_NUM_VERTS] __attribute__((aligned(32)));

uint32 norm_to_argb(vec3f_t norm) {
  uint8_t r = (int)((0.5f * norm.x + 0.5f) * 255.0f);
  uint8_t g = (int)((0.5f * norm.y + 0.5f) * 255.0f);
  uint8_t b = (int)((0.5f * norm.z + 0.5f) * 255.0f);
  return ((0xFF<<24) | (r<<16) | (g<<8) | b);
}

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

vec3f_t light = {0.0f, 0.0f, -1.0f};
vec3f_t light_col = {0.5f, 0.5f, 0.5f};

void render_cloth(void) {
  mat_load(&stored_projection_view);
  mat_translate(cube_state.pos.x, cube_state.pos.y, cube_state.pos.z);
  mat_scale(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE);
  mat_rotate_x(cube_state.rot.x);
  mat_rotate_y(cube_state.rot.y);


  mat_transform((vector_t*)cur_cloth_verts, (vector_t*)&trans_verts, CLOTH_TOTAL_NUM_VERTS, sizeof(vec3f_t));

  float ambient = 0.6f;
  for(int y = 0; y < CLOTH_NUM_VERTS_ONE_AXIS; y++) {
    int y_off = (y*CLOTH_NUM_VERTS_ONE_AXIS);

    for(int x = 0; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {
      vec3f_t norm = cloth_normals[y_off+x];

      //printf("%f -> %f, %f -> %f\n", u0, u1, v0, v1);
      float flt_r = 0.0f;
      float flt_g = 0.0f;
      float flt_b = 0.0f;
      float flt_intensity = fipr(light.x, light.y, light.z, 0.0f, norm.x, norm.y, norm.z, 0.0f);
      
        flt_r += (flt_intensity * light_col.x);
        flt_g += (flt_intensity * light_col.y);
        flt_b += (flt_intensity * light_col.z);

        flt_r = min(max(0.0f, flt_r), 1.0f-ambient);// / 8.0f;
        flt_g = min(max(0.0f, flt_g), 1.0f-ambient);// / 4.0f;
        flt_b = min(max(0.0f, flt_b), 1.0f-ambient);// / 8.0f;
        flt_r += ambient;
        flt_g += ambient;
        flt_b += ambient;
        //flt_r = gamma_correct(flt_r);
        //flt_g = gamma_correct(flt_g);
        //flt_b = gamma_correct(flt_b);

        
        uint8_t r_intensity = (flt_r)*255.0f;
        uint8_t g_intensity = (flt_g)*255.0f;
        uint8_t b_intensity = (flt_b)*255.0f;

        cloth_vert_argbs[y_off+x] = (0x80<<24) | (r_intensity << 16) | (g_intensity << 8) | b_intensity;
    }
  }
            


  pvr_poly_cxt_t cxt;
  pvr_dr_state_t dr_state;
  pvr_poly_hdr_t hdr;

  init_cloth_context(&cxt);
  pvr_poly_compile(&hdr, &cxt);
  pvr_prim(&hdr, sizeof(hdr));
  pvr_dr_init(&dr_state);

  
  pvr_vertex_t *vert;

  for(int y = 0; y < CLOTH_NUM_POLYS_ONE_AXIS; y++) {
    int ny = y+1;
    int y_off = (y*CLOTH_NUM_VERTS_ONE_AXIS);
    int dy_off = (ny*CLOTH_NUM_VERTS_ONE_AXIS);
    float uv = ((float)y)/CLOTH_NUM_VERTS_ONE_AXIS;
    float dv = ((float)ny)/CLOTH_NUM_VERTS_ONE_AXIS;

    float ux = trans_verts[y_off].x;
    float uy = trans_verts[y_off].y;
    float uz = trans_verts[y_off].z;
    float dx = trans_verts[dy_off].x;
    float dy = trans_verts[dy_off].y;
    float dz = trans_verts[dy_off].z;
    uint32 ucol = cloth_vert_argbs[y_off];
    uint32 dcol = cloth_vert_argbs[dy_off];

    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX;
    vert->x = ux;
    vert->y = uy;
    vert->z = uz;
    vert->u = 0.0f;
    vert->v = uv;
    vert->argb = ucol;
    vert->oargb = 0x00000000;
    pvr_dr_commit(vert);
    vert = pvr_dr_target(dr_state);
    vert->flags = PVR_CMD_VERTEX;
    vert->x = dx;
    vert->y = dy;
    vert->z = dz;
    vert->u = 0.0f;
    vert->v = dv;
    vert->argb = dcol;
    vert->oargb = 0x00000000;
    pvr_dr_commit(vert);



    for(int x = 1; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {
      float ux = trans_verts[y_off+x].x;
      float uy = trans_verts[y_off+x].y;
      float uz = trans_verts[y_off+x].z;
      float dx = trans_verts[dy_off+x].x;
      float dy = trans_verts[dy_off+x].y;
      float dz = trans_verts[dy_off+x].z;

      float u = ((float)x)/CLOTH_NUM_VERTS_ONE_AXIS;
      uint32 ucol = cloth_vert_argbs[y_off+x];
      uint32 dcol = cloth_vert_argbs[dy_off+x];


      vert = pvr_dr_target(dr_state);
      vert->flags = PVR_CMD_VERTEX;
      vert->x = ux;
      vert->y = uy;
      vert->z = uz;
      vert->u = u;
      vert->v = uv;
      vert->argb = ucol;
      vert->oargb = 0x00000000;
      pvr_dr_commit(vert);

      vert = pvr_dr_target(dr_state);
      vert->flags = x == CLOTH_NUM_VERTS_ONE_AXIS-1 ? PVR_CMD_VERTEX_EOL : PVR_CMD_VERTEX;
      vert->x = dx;
      vert->y = dy;
      vert->z = dz;
      vert->u = u;
      vert->v = dv;
      vert->argb = dcol;
      vert->oargb = 0x00000000;
      pvr_dr_commit(vert);
    }
  }
  pvr_dr_finish();

}

static inline void cube_reset_state() {
  cube_state = (struct cube){0};
  fovy = DEFAULT_FOV;
  cube_state.pos.z = (MAX_ZOOM + MIN_ZOOM) / 2.0f;
  cube_state.rot.x = 0.3f;
  cube_state.rot.y = 0.5f;
  update_projection_view(fovy);
}

float lerp(float a, float b, float f) {
    return a * (1.0 - f) + (b * f);
}


vec3f_t cross(vec3f_t a, vec3f_t b) {
  vec3f_t res;
  res.x = (a.y*b.z) - (a.z*b.y);
  res.y = (a.z*b.x) - (a.x*b.z);
  res.z = (a.x*b.y) - (a.y*b.x);
  return res;
}
vec3f_t vec3f_sub(vec3f_t a, vec3f_t b) {
  vec3f_t res;
  res.x = a.x-b.x;
  res.y = a.y-b.y;
  res.z = a.z-b.z;
  return res;
}

vec3f_t get_normal(int y_off, int dy_off, int px) {
  int rx = px == CLOTH_NUM_VERTS_ONE_AXIS-1 ? px : px+1;
  int l_idx = y_off+px;
  int dr_idx = dy_off+rx;
  int r_idx = y_off+rx;
  int dn_idx = dy_off+px;


  vec3f_t a = vec3f_sub(cur_cloth_verts[dr_idx], cur_cloth_verts[r_idx]);
  vec3f_t b = vec3f_sub(cur_cloth_verts[dn_idx], cur_cloth_verts[l_idx]);

  vec3f_t normal = cross(a, b);
  vec3f_normalize(normal.x, normal.y, normal.z);
  return normal;
}

void recalc_normals() {

  vec3f_t default_normal = {.x = 0, .y = 0, .z = -1.0f};
  for(int py = 0; py < CLOTH_NUM_VERTS_ONE_AXIS; py++) {
    int up_y = py == 0 ? py : py-1;
    int dn_y = py == CLOTH_NUM_VERTS_ONE_AXIS-1 ? py : py+1; 
    int uy_off = (up_y * CLOTH_NUM_VERTS_ONE_AXIS);
    int y_off = (py * CLOTH_NUM_VERTS_ONE_AXIS);
    int dy_off = dn_y * CLOTH_NUM_VERTS_ONE_AXIS;

    vec3f_t prev_up_norm = {.x = 0.0f, .y = 0.0f, .z = -1.0f};
    vec3f_t prev_norm = {.x = 0.0f, .y = 0.0f, .z = -1.0f};

    //vec3f_t l = {.x = 0, .y = 0, .z = -1.0f};
    for(int px = 0; px < CLOTH_NUM_VERTS_ONE_AXIS; px++) {
      //int lx = px == 0 ? px : px-1;

      //vec3f_t nl = get_normal(uy_off, y_off, lx);
      
      //vec3f_t nr = get_normal(uy_off, y_off, px);
      //vec3f_t ndl = get_normal(y_off, dy_off, lx);
      //vec3f_t ndr = get_normal(y_off, dy_off, px);
      vec3f_t up_norm = (py == 0) ? default_normal : cloth_normals[uy_off+px];
      vec3f_t normal = get_normal(y_off, dy_off, px);

      float avg_x = (normal.x + prev_norm.x + up_norm.x) / 3.0f;
      float avg_y = (normal.y + prev_norm.y + up_norm.y) / 3.0f;
      float avg_z = (normal.z + prev_norm.z + up_norm.z) / 3.0f;
      vec3f_normalize(avg_x, avg_y, avg_z);
      //cloth_normals[y_off+px] = normal;
      cloth_normals[y_off+px].x = avg_x;
      cloth_normals[y_off+px].y = avg_y;
      cloth_normals[y_off+px].z = avg_z;
      prev_norm = normal;
      //prev_up_norm = up_norm;
      //cloth_normals[y_off+px].x = (nl.x + nr.x + ndl.x + ndr.x) / 4.0f;
      //cloth_normals[y_off+px].y = (nl.y + nr.y + ndl.y + ndr.y) / 4.0f;
      //cloth_normals[y_off+px].z = (nl.z + nr.z + ndl.z + ndr.z) / 4.0f;

    }
  }
}

#define NUM_SEGMENTS  (((CLOTH_NUM_VERTS_ONE_AXIS - 1) * CLOTH_NUM_VERTS_ONE_AXIS) * 2)


void reset_cloth() {
  cur_cloth_verts = cloth_verts1;
  next_cloth_verts = cloth_verts2;
  float min_extent = -1.0f;
  float max_extent = 1.0f;

  float z_off_at_bottom[CLOTH_NUM_VERTS_ONE_AXIS];
  for(int x = 0; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {
      int rand_to_32 = rand() & 31;
      float z_off = lerp(-.05f, .05f, rand_to_32/32.0f);
      z_off_at_bottom[x] = z_off;
  }

  for(int y = 0; y < CLOTH_NUM_VERTS_ONE_AXIS; y++) {
    int can_lock_y = y == 0;
    float fy = lerp(min_extent, max_extent, ((float)y)/CLOTH_NUM_VERTS_ONE_AXIS);

    float percent_to_bottom = ((float)y) / CLOTH_NUM_VERTS_ONE_AXIS;

    for(int x = 0; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {


      float z_off = lerp(0.0f, z_off_at_bottom[x], percent_to_bottom);


      float fx = lerp(min_extent, max_extent, ((float)x)/CLOTH_NUM_VERTS_ONE_AXIS);
      bool can_lock_x = (x % 4) == 0;

      int idx = y*CLOTH_NUM_VERTS_ONE_AXIS+x;
      cur_cloth_verts[idx].x = fx;
      cur_cloth_verts[idx].y = fy;
      cur_cloth_verts[idx].z = z_off;
      next_cloth_verts[idx].x = fx;
      next_cloth_verts[idx].y = fy;
      next_cloth_verts[idx].z = z_off;
      cloth_prev_verts[idx].x = fx;
      cloth_prev_verts[idx].y = fy;
      cloth_prev_verts[idx].z = z_off;

      point_is_locked[idx] = (can_lock_x && can_lock_y) ? 1 : 0;
    }
  }

  
  int segment = 0;
  for(int x = 0; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {
    for(int y = 0; y < CLOTH_NUM_VERTS_ONE_AXIS; y++) {
      int idx = (y*CLOTH_NUM_VERTS_ONE_AXIS)+x;
      int d_idx = ((y+1)*CLOTH_NUM_VERTS_ONE_AXIS)+x;
      if(x < CLOTH_NUM_VERTS_ONE_AXIS-1) {
        segments[segment].fst = idx;
        segments[segment++].snd = idx+1;
      }
      if(y < CLOTH_NUM_VERTS_ONE_AXIS-1) {
        segments[segment].fst = idx;
        segments[segment++].snd = d_idx;
      }
    }
  }
}

void drop_pickup_cloth() {
  for(int vert_y = 0; vert_y < CLOTH_NUM_VERTS_ONE_AXIS; vert_y++) {
        bool can_lock_y = vert_y == 0;

        for(int vert_x = 0; vert_x < CLOTH_NUM_VERTS_ONE_AXIS; vert_x++) {
            bool can_lock_x = vert_x % 4 == 0;

            // locked -> not locked
            // not_locked && can_lock -> locked

            if(point_is_locked[vert_y*CLOTH_NUM_VERTS_ONE_AXIS+vert_x]) {
              point_is_locked[vert_y*CLOTH_NUM_VERTS_ONE_AXIS + vert_x] = 0;
            } else if (can_lock_x && can_lock_y) {
              point_is_locked[vert_y*CLOTH_NUM_VERTS_ONE_AXIS+vert_x] = 1;
            }
        }
    }
}

void do_gravity() {
  for(int y = 0; y < CLOTH_NUM_VERTS_ONE_AXIS; y++) {
    int y_off = y*CLOTH_NUM_VERTS_ONE_AXIS;
    for(int x = 0; x < CLOTH_NUM_VERTS_ONE_AXIS; x++) {
      int idx = y_off + x;
      if(!point_is_locked[idx]) {
        float prev_x = cur_cloth_verts[idx].x;
        float prev_y = cur_cloth_verts[idx].y;
        float prev_z = cur_cloth_verts[idx].z;
      
        cur_cloth_verts[idx].x += (cur_cloth_verts[idx].x - cloth_prev_verts[idx].x) + gravity.x;
        cur_cloth_verts[idx].y += (cur_cloth_verts[idx].y - cloth_prev_verts[idx].y) + gravity.y;
        cur_cloth_verts[idx].z += (cur_cloth_verts[idx].z - cloth_prev_verts[idx].z) + gravity.z;
        cloth_prev_verts[idx].x = prev_x;
        cloth_prev_verts[idx].y = prev_y;
        cloth_prev_verts[idx].z = prev_z;
      }
    }
  }
}


void do_sim(int cur_frame) {  

  for(int segment = 0; segment < NUM_SEGMENTS; segment += 1) {
      int fst_idx = segments[segment].fst;
      int snd_idx = segments[segment].snd;
      int fst_locked = point_is_locked[fst_idx];
      int snd_locked = point_is_locked[snd_idx];

      float v1x = cur_cloth_verts[fst_idx].x;
      float v1y = cur_cloth_verts[fst_idx].y;
      float v1z = cur_cloth_verts[fst_idx].z;
      float v2x = cur_cloth_verts[snd_idx].x;
      float v2y = cur_cloth_verts[snd_idx].y;
      float v2z = cur_cloth_verts[snd_idx].z;


      // horizontal stretch correction
      
      float dirx = v1x-v2x;
      float diry = v1y-v2y;
      float dirz = v1z-v2z;
      
      float inv_seg_len = frsqrt(dirx*dirx+diry*diry+dirz*dirz);
      float seg_len = 1.0f / inv_seg_len;

      if(seg_len > CLOTH_SPACING) {
        float scale = CLOTH_SPACING / seg_len;

        float cenx = (v1x + v2x) / 2.0f;
        float ceny = (v1y + v2y) / 2.0f;
        float cenz = (v1z + v2z) / 2.0f;
        float segx = dirx * scale / 2.0f;
        float segy = diry * scale / 2.0f;
        float segz = dirz * scale / 2.0f;


        if(!fst_locked) {
          v1x = cenx + segx;
          v1y = ceny + segy;
          v1z = cenz + segz;
          cur_cloth_verts[fst_idx].x = v1x;
          cur_cloth_verts[fst_idx].y = v1y;
          cur_cloth_verts[fst_idx].z = v1z;
        }
        
        if(!snd_locked) {
          v2x = cenx - segx;
          v2y = ceny - segy;
          v2z = cenz - segz;
          cur_cloth_verts[snd_idx].x = v2x;
          cur_cloth_verts[snd_idx].y = v2y;
          cur_cloth_verts[snd_idx].z = v2z;
        }
      }
    }
  recalc_normals();
 
}



int update_state(int cur_frame) {
  do_gravity();
  //vec3f_t* tmp_verts = cur_cloth_verts;
  //cur_cloth_verts = next_cloth_verts;
  //next_cloth_verts = tmp_verts;
  do_sim(cur_frame);
  //do_sim(1, 0);
  //do_sim(0, 1);
  //do_sim(1, 1);
  //vec3f_t* tmp_verts = cur_cloth_verts;
  //cur_cloth_verts = next_cloth_verts;
  //next_cloth_verts = tmp_verts;

  static int last_a = 0;
  int keep_running = 1;
  MAPLE_FOREACH_BEGIN(MAPLE_FUNC_CONTROLLER, cont_state_t, state)
  if (state->buttons & CONT_START){
    keep_running = 0;
  }

  //if (abs(state->joyx) > 16)
  //  cube_state.pos.x += (state->joyx / 32768.0f) * 20.5f; // Increased sensitivity
  //if (abs(state->joyy) > 16)
  //  cube_state.pos.y += (state->joyy / 32768.0f) * 20.5f; // Increased sensitivity and inverted Y
  

  if (state->ltrig > 16) // Left trigger to zoom out
    cube_state.pos.z -= (state->ltrig / 255.0f) * ZOOM_SPEED;
  if (state->rtrig > 16) // Right trigger to zoom in
    cube_state.pos.z += (state->rtrig / 255.0f) * ZOOM_SPEED;
  if (cube_state.pos.z < MIN_ZOOM)
    cube_state.pos.z = MIN_ZOOM; // Farther away
  if (cube_state.pos.z > MAX_ZOOM)
    cube_state.pos.z = MAX_ZOOM; // Closer to the screen

  cube_state.rot.y += lerp(-0.07f, 0.07f, ((state->joyx + 128) / 256.0f));
  cube_state.rot.x += lerp(-0.07f, 0.07f, ((state->joyy + 128) / 256.0f));

  //if (state->buttons & CONT_X) {
  //  cube_state.speed.y += 0.001f;
  //}

  //if (state->buttons & CONT_B) {
  //  cube_state.speed.y -= 0.001f;
  //}

  if (state->buttons & CONT_A) {
    if(!last_a) {
      printf("drop/pickup\n");
      drop_pickup_cloth();
    }
    last_a = 1;
  } else {
    last_a = 0;
    //cube_state.speed.x += 0.001f;
  }

  //if (state->buttons & CONT_Y) {
  //  //cube_state.speed.x -= 0.001f;
  //}
  static int prev_left = 0;
  if (state->buttons & CONT_DPAD_LEFT) {
    //cube_state = (struct cube){0};
    //fovy = DEFAULT_FOV;
    //cube_reset_state();
    if(!prev_left) {
      reset_cloth();
    }
    prev_left = 1;
  } else {
    prev_left = 0;
  }
  if (state->buttons & CONT_DPAD_DOWN) {
    fovy -= 1.0f;
    update_projection_view(fovy);
  }
  if (state->buttons & CONT_DPAD_UP) {
    fovy += 1.0f;
    update_projection_view(fovy);
  }

  if (state->buttons & CONT_DPAD_RIGHT) {
    printf("fovy = %f\n"
           "cube_state.pos.x = %f\n"
           "cube_state.pos.y = %f\n"
           "cube_state.pos.z = %f\n"
           "cube_state.rot.x = %f\n"
           "cube_state.rot.y = %f\n"
           "cube_state.speed.x = %f\n"
           "cube_state.speed.y = %f\n",
           fovy, cube_state.pos.x, cube_state.pos.y, cube_state.pos.z,
           cube_state.rot.x, cube_state.rot.y, cube_state.speed.x,
           cube_state.speed.y);
  }
  MAPLE_FOREACH_END()

  // Apply rotation
  //cube_state.rot.x += cube_state.speed.x;
  //cube_state.rot.y += cube_state.speed.y;

  // Apply friction
  //cube_state.speed.x *= 0.99f;
  //cube_state.speed.y *= 0.99f;
  return keep_running;
}

void swap_random_segments() {
    return;
    int x = rand()%NUM_SEGMENTS;
    int y = rand()%NUM_SEGMENTS;
    while(x == y) {
      x = rand()%NUM_SEGMENTS;
      y = rand()%NUM_SEGMENTS;
    }
    segment tmp = segments[x];
    segments[x] = segments[y];
    segments[y] = tmp;
}

int main(int argc, char *argv[]) {
  
  srand(time(NULL));

  pvr_init_params_t params = {
    // opaque, opaque modifiers, translucent, translucent modifiers, punch-thru
      {PVR_BINSIZE_16, PVR_BINSIZE_0, PVR_BINSIZE_16, PVR_BINSIZE_0,
       PVR_BINSIZE_0},
      1024 * 1024, // Vertex buffer size
      0, // No DMA
      0, //  No FSAA
      0,  // Translucent Autosort enabled.
      .opb_overflow_count = 3,
  };

  pvr_init(&params);
  pvr_set_bg_color(135.0f/255.0f, 206.0f/255.0f, 235.0f/255.0f);

  if (!pvrtex_load("/rd/texture/rgb565_vq_tw/dc.dt", &texture)) {
    printf("Failed to load texture.\n");
    return -1;
  }
  reset_cloth();
  cube_reset_state();

  for(int i = 0; i < CLOTH_NUM_VERTS_ONE_AXIS; i++) {
    swap_random_segments();
  }

  uint32 cur_frame = 0;
  while (1) {
    uint64 time_start = timer_us_gettime64();

    swap_random_segments();
    if (!update_state(cur_frame)) {
      break;
    }
    uint64 update_time_end = timer_us_gettime64();
    uint64 update_dtime = update_time_end - time_start;
    printf("sim time: %f\n", update_dtime/1000.0f);


    pvr_wait_ready();
    pvr_scene_begin();


    pvr_list_begin(PVR_LIST_TR_POLY);


    float frame_zero_to_one = (cur_frame & 1023) / 1024.0f;
    float rads = lerp(0.0f, 2.0f*3.14159f, frame_zero_to_one);
    //printf("rads %f\n", light.z);
    light.z = fsin(rads);
    //printf("light.z %f\n", light.z);
    cur_frame++;

    render_cloth();
    pvr_list_finish();

    pvr_scene_finish();
    uint64 time_end = timer_us_gettime64();
    uint64 dtime = time_end - time_start;
    printf("total time: %f\n", dtime/1000.0f);

  }

  printf("Cleaning up\n");
  pvrtex_unload(&texture);
  pvr_shutdown(); // Clean up PVR resources
  vid_shutdown(); // This function reinitializes the video system to what dcload
                  // and friends expect it to be Run the main application here;

  printf("Exiting main\n");
  return 0;
}