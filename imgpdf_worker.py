import streamlit as st 
import pandas as pd 
from pdf2image import convert_from_bytes
import easyocr
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import json
import copy
from io import BytesIO
import time

st.set_page_config(
    page_title='pdf_converter',
    layout='wide',
    # Theme settings
    initial_sidebar_state="expanded"
)
#%% Initialize variables in the session state
if not "config" in st.session_state:
    st.session_state["config"]={
        "characters_allowed":"$0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ.,%-/:|"
        ,"rotate_degrees":0
        ,"languages":["en"]
        ,"scale_x":2
        ,"scale_y":2
        ,"pdf_path":'C:\\Users\\Barid.Temple\\Anaconda\\envs\\default\\project_files\\easyOCR\\Paccar_Returns\\Mississauga December Return.pdf'
        ,"poppler_path":"C:\\Users\\Barid.Temple\\Anaconda\\envs\\default\\project_files\\easyOCR\\poppler-23.11.0\\Library\\bin"
    }
if not "ocr_reader" in st.session_state:
    st.session_state["ocr_reader"] = easyocr.Reader(st.session_state["config"]["languages"])
if not "current_page" in st.session_state:
    st.session_state["current_page"]="load_pdf"
if not "images" in st.session_state:
    st.session_state["images"]=None
if not "current_image" in st.session_state:
    st.session_state["current_image"] = 0

if not "templates" in st.session_state:
    st.session_state["templates"] = {"Default":{}}

if not "x_loc" in st.session_state:
    st.session_state["x_loc"]= 0
    st.session_state["y_loc"]= 0

if not "out_data" in st.session_state:
    st.session_state["out_data"] = {}


#%% Load items in the sidebar
with st.sidebar:
    st.write("Converter Settings")
    
    with st.expander("loading_options"):
        language = st.selectbox("Language",options=st.session_state["config"]["languages"])
        #chars_allowed = st.text_input("characters_allowed",st.session_state["config"]["characters_allowed"])
        rotation_degrees = st.text_input("rotate_degrees",st.session_state["config"]["rotate_degrees"])
        scale_x = st.text_input("scale_x",st.session_state["config"]["scale_x"])
        scale_y = st.text_input("scale_y",st.session_state["config"]["scale_y"])
        poppler_path = st.text_input("poppler_path",st.session_state["config"]["poppler_path"])
        back_button = st.button("back")
        if back_button:
            st.session_state["current_page"]="load_pdf"
            st.rerun()
    
    if st.session_state["images"]!=None:
    
        with st.expander("page_navigation"):
            col1,col2 = st.columns(2)
            imsize = st.number_input("width",step=100,format="%i")
            current_image = st.text_input("current_image",0)
            select_image = st.button("Go")
            current_image = int(float(current_image))
            previous_image = col1.button(" \<\< ")
            next_image = col2.button(" \>\> ")

    
        with st.expander("template_builder"):
            current_template = st.text_input("template","Default")
            variable_name = st.text_input("variable_name")
            chars_allowed = st.text_input("characters_allowed", key="characters_allowed", value="")
            
            st.write(f"Dimensions: {st.session_state['images'][st.session_state['current_image']].shape[1] - 1} x {st.session_state['images'][st.session_state['current_image']].shape[0] - 1}")
            
            x_loc =st.select_slider("X",options=list(np.arange(0,int(st.session_state['images'][st.session_state['current_image']].shape[1]))),value=st.session_state["x_loc"])
            y_loc =st.select_slider("Y",options=list(np.arange(0,int(st.session_state['images'][st.session_state['current_image']].shape[0]))),value=st.session_state["y_loc"])
            
            width = st.number_input("width",step=10,format="%i")
            height = st.number_input("height",step=10,format="%i")

            x_jump = st.number_input("x_jump",step=10,format="%i")
            y_jump = st.number_input("y_jump",step=10,format="%i")
            
            jmp,jump_set = st.columns(2)
            jump_button =jmp.button("jump")
            j_set = jump_set.button("jump_set")
            if jump_button:
                x_loc+=x_jump
                y_loc+=y_jump
                st.session_state["x_loc"] = x_loc
                st.session_state["y_loc"] = y_loc
            if j_set:
                x_loc+=x_jump
                y_loc+=y_jump
                st.session_state["x_loc"] = x_loc
                st.session_state["y_loc"] = y_loc
            colset,coldel = st.columns(2)
            set_image = colset.button("set")
            del_ind = coldel.button("delete")
            x0 = int(x_loc)
            x1 = int(x_loc+width)
            y0 = int(y_loc)
            y1 = int(y_loc+height)


        with st.expander("template_upload"):
            uploaded_name = st.text_input("Template Name")
            upload_template = st.file_uploader("Upload Template")
            upload_template_button = st.button("Import")

            if upload_template_button:
                st.session_state["templates"][uploaded_name] = json.load(upload_template)


        with st.expander("template_export"):
            selected_export_template = st.selectbox("Template",options=list(st.session_state["templates"].keys()))
            export_template = st.download_button("Export",data=json.dumps(st.session_state["templates"][selected_export_template],indent=1),file_name=selected_export_template+".json")

        with st.expander("apply_template"):
            selected_apply_template = st.selectbox("template",options=list(st.session_state["templates"].keys()))
            selected_pages = st.text_input("pages")
            apply_template = st.button("apply")

        
        


#%% Create a landing page to load a pdf file
if st.session_state["current_page"]=="load_pdf":
    st.title("Image PDF Converter")
    pdf_file = st.file_uploader("Please Select a File")

    if pdf_file:
        with st.spinner("Finding Images"):
            pdf_bytes = pdf_file.getvalue()
            st.session_state["images"] = convert_from_bytes(pdf_bytes,poppler_path =r""+poppler_path)

            for i, image in enumerate(st.session_state["images"]):
                st.session_state["images"][i] = st.session_state["images"][i].rotate(float(rotation_degrees),expand=True)
                st.session_state["images"][i] = np.array(st.session_state["images"][i])
                st.session_state["images"][i] = cv2.resize(
                    st.session_state["images"][i],
                    None,
                    fx=int(scale_x),
                    fy=int(scale_y),
                    interpolation=cv2.INTER_LINEAR
                )
        if st.session_state["images"]!=None:
            st.session_state["current_page"]="main"
            st.rerun()
        else:
            st.error("No Images Found")



def add_box(image,bound_box):
    try:
        x_0, x_1 = max(min(int(bound_box["x0"]), int(bound_box["x1"])), 0), min(max(int(bound_box["x0"]), int(bound_box["x1"])), image.shape[1] - 1)
        y_0, y_1 = max(min(int(bound_box["y0"]), int(bound_box["y1"])), 0), min(max(int(bound_box["y0"]), int(bound_box["y1"])), image.shape[0] - 1)
        thickness = 5  # Example thicknes
        image[y_0:y_0+thickness, x_0:x_1] = [255, 0, 0]  # Top border
        image[y_1:y_1+thickness, x_0:x_1] = [255, 0, 0]  # Bottom border
        image[y_0:y_1, x_0:x_0+thickness] = [255, 0, 0]  # Left border
        image[y_0:y_1, x_1:x_1+thickness] = [255, 0, 0]  # Right border
        return image
    except:
        st.error("Please Create a Valid Bounding Box")

if st.session_state["current_page"]=="main":

    #st.title("Image PDF Converter")

    image = st.session_state["images"][st.session_state["current_image"]].copy()

    if x_loc or y_loc or width or height:
        try:
            x0, x1 = max(min(int(x0), int(x1)), 0), min(max(int(x0), int(x1)), image.shape[1] - 1)
            y0, y1 = max(min(int(y0), int(y1)), 0), min(max(int(y0), int(y1)), image.shape[0] - 1)
            # Border thickness
            thickness = 5  # Example thickness

            # Highlighting the bounding box in red
            # Top and bottom borders
            image[y0:y0+thickness, x0:x1] = [255, 0, 0]  # Top border
            image[y1:y1+thickness, x0:x1] = [255, 0, 0]  # Bottom border

            # Left and right borders
            image[y0:y1, x0:x0+thickness] = [255, 0, 0]  # Left border
            image[y0:y1, x1:x1+thickness] = [255, 0, 0]  # Right border
        except:
            st.error("Please Create a Valid Bounding Box")

    chars_allowed = st.session_state["config"]["characters_allowed"] if chars_allowed == "" else chars_allowed
    if set_image:
        try:
            x0, x1 = max(min(int(x0), int(x1)), 0), min(max(int(x0), int(x1)), image.shape[1] - 1)
            y0, y1 = max(min(int(y0), int(y1)), 0), min(max(int(y0), int(y1)), image.shape[0] - 1)
            # Border thickness
            thickness = 5  # Example thickness

            if current_template not in st.session_state["templates"]:
                st.session_state["templates"][current_template]={}
            if not variable_name in st.session_state["templates"][current_template]:
                st.session_state["templates"][current_template][variable_name] = []
                
            st.session_state["templates"][current_template][variable_name].append({"x0":x0,"x1":x1,"y0":y0,"y1":y1,"chars_allowed":chars_allowed})
        except:
            st.error("Please Create a Valid Bounding Box")

    if j_set:
        try:
            x0, x1 = max(min(int(x0), int(x1)), 0), min(max(int(x0), int(x1)), image.shape[1] - 1)
            y0, y1 = max(min(int(y0), int(y1)), 0), min(max(int(y0), int(y1)), image.shape[0] - 1)
            # Border thickness
            thickness = 5  # Example thickness
            
            if current_template not in st.session_state["templates"]:
                st.session_state["templates"][current_template]={}
            if not variable_name in st.session_state["templates"][current_template]:
                st.session_state["templates"][current_template][variable_name] = []
                
            st.session_state["templates"][current_template][variable_name].append({"x0":x0,"x1":x1,"y0":y0,"y1":y1,"chars_allowed":chars_allowed})

        except:
            st.error("Please Create a Valid Bounding Box")

    if del_ind:
        try:
            if len(st.session_state["templates"][current_template][variable_name])==0:
                del(st.session_state["templates"][current_template][variable_name])
            else:
                st.session_state["templates"][current_template][variable_name].pop(len(st.session_state["templates"][current_template][variable_name])-1)
        except:
            st.error("Variable Does Not Exist")


    if current_template in st.session_state["templates"]:
        for key in list(st.session_state["templates"][current_template].keys()):
            for box in st.session_state["templates"][current_template][key]:
                add_box(image,box)

    
    if imsize>100:
        st.image(image,width=imsize)
    else:
        st.image(image)
    st.write("current_template")
    st.write(current_template)
    json_objects =  st.container()
    if current_template in st.session_state["templates"]:
        json_objects.json(json.dumps(st.session_state["templates"][current_template],indent=1),expanded=False)


    if select_image:
        if current_image<=len(st.session_state["images"])-1:
            st.session_state["current_image"] = current_image
            st.rerun()
        else: st.warning("Please select a valid indice")
    
    if next_image:
        if (st.session_state["current_image"]+1)<=len(st.session_state["images"])-1:
            st.session_state["current_image"] = st.session_state["current_image"]+1
            st.rerun()
    if previous_image:
        if st.session_state["current_image"]>=1:
            st.session_state["current_image"] = st.session_state["current_image"]-1
            st.rerun()



    if apply_template:
        with st.spinner("predicting and structuring bounding boxes"):
            pages_string = selected_pages
            pages_list = str.split(pages_string,sep=",")
            pages = []
            for i, item in enumerate(pages_list):
                page_min = int(str.split(item,sep="-")[0])
                try:
                    page_max = int(str.split(item,sep="-")[1])
                except:
                    page_max = int(str.split(item,sep="-")[0])
                pages = pages + list(np.arange(page_min,page_max+1))
            boxes = st.session_state["templates"][selected_apply_template]

            pages.sort()
            pages = list(set(pages))
            
            outdata = {}
            progress_text = ""
            prog = st.progress(0, text=progress_text)
            start = time.time()
            single_page = 0
            for p ,pagenum in enumerate(pages):
                time.sleep(0.01)
                duration = round((time.time()-start)/60,2)
                perc_cplt = (p)/len(pages)
                if single_page == 0:
                    single_page = duration
                    estimate = round((len(pages)-p)*single_page,2)
                else:
                    estimate = round((len(pages)-p)*.5,2)
                prog.progress(perc_cplt, text=f"prog: {perc_cplt*100}%, page: {pagenum}, time: {duration} min, est: {estimate} min")
                for key in list(boxes.keys()):
                    if not key in outdata:
                        outdata[key] = []
                    for i, box in enumerate(boxes[key]):
                        cropped_image = st.session_state['images'][pagenum][int(box["y0"]):int(box["y1"]),int(box["x0"]):int(box["x1"])]
                        prediction = st.session_state["ocr_reader"].readtext(cropped_image,allowlist=box["chars_allowed"])
                        predictions = " ".join([pred[-2] for pred in prediction])
                        outdata[key].append(predictions)
                        #st.write(predictions)
                        #outdata[key].append[predictions]
            st.session_state["out_data"] = outdata
            
            excel_buffer = BytesIO()
            df = pd.DataFrame(st.session_state["out_data"].values())
            df = df.T
            df.columns = list(boxes.keys())
            df.to_excel(excel_buffer,index=False)
            excel_buffer.seek(0)

            st.session_state["excel_buffer"] = excel_buffer
            st.session_state["out_data"] = df
            st.rerun()

    
    if "out_data" in st.session_state:
        edit_data = st.data_editor(st.session_state["out_data"],height=800,)
        try:    
            apply_changes = st.button("apply_changes")
            if apply_changes:
                excel_buffer = BytesIO()
                df = edit_data
                df.to_excel(excel_buffer,index=False)
                excel_buffer.seek(0)
                st.session_state["excel_buffer"] = excel_buffer
                st.session_state["out_data"] = df

            st.download_button(label="Download Excel File", data=st.session_state["excel_buffer"], file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except:
            pass

            
# %%
