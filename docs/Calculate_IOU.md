# IVIT-I Application of Calculate IOU
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
|application(*)|dict|{  }|TEncapsulating all information of configuration.|
|depend_on(*)|list|[ ]|The object that you want to calculate iou.  |
|palette|list|[ ]|The object that you want to change background. |


* Basic sample
    ```json
        {
            "application": {
                "palette": {
                    "car": [
                        0,
                        0,
                        0
                    ]
                },
                "areas": [
                    {
                        "name": "default",
                        "depend_on": [],
                    }
                ],
            }
        }

    ```
## Create Instance
You need to use [app_config](#setting-app-config) and label path to create instance of application.
   ```python
    from apps import Calculate_IOU

    app = Calculate_IOU( app_config, label_path )

   ``` 
  
   
## Format of input parameter
* Input parameters are the result of model predict, and the type of result must be np.ndarray and shape( width and hight) must same with the original image.


## Application output 
* Return treated frame.
