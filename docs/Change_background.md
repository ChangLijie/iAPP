# IVIT-I Application of Change background
## Usage
You need to follow the step below to use application:  
Step 1. [Setting Config](#setting-app-config).  
Step 2. [Create Instance](#create-instance).  
Step 3. Follow the [format of input parameter](#format-of-input-parameter) to use application.  
And the description of application output is [here](#application-output).
## Setting app config 
* The description of key from config.(*) represent must be set.  

| Name | Type | Default | Description |
| --- | --- | --- | --- |
|application(*)|dict|{  }|Encapsulating all information of configuration.|
|background_frame(*)|str|“ ”|Seting the location of detection area. |
|change_object（*）|list|[ ]|The object that you want to change background.|


* Basic sample
    ```json
        {
            "application": {
                "background_frame":"/workspace/data/background.jpg",
                "change_object":[]
            }
        }

    ```
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    from apps import Change_background

    app = Change_background( app_config, label_path )

   ``` 
  
   
## Format of input parameter
* Input parameters are the result of model predict, and the type of result must be np.ndarray and shape( width and hight) must same with the original image.


## Application output 
* Return treated frame.
