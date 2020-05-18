package cz.ivoa.vocloud.rest;

import javax.ws.rs.*;
import java.io.FileWriter;
import java.io.IOException;
import javax.faces.context.FacesContext;

/**
 * Created by tma on 12.4.20
 */

@Path("/labelData")
public class LabelDataUpload {
    String returnStr;
    @Path("/upload")
    @POST
    @Consumes("text/plain")
    @Produces("text/plain")
    public String postMessage (@QueryParam("data") String data, @QueryParam("path") String path) {
        String completePath = "/var/local/vocloud/filesystem/DATA/"+path;
        try {
            FileWriter writer = new FileWriter(completePath, false);
            writer.write(data);
            writer.close();	
			String idParam = FacesContext.getCurrentInstance().getExternalContext().getRequestParameterMap().get("jobId");
            returnStr=idParam;
        } catch (IOException e) {
            e.printStackTrace();
            returnStr="notOK";
        }
        return returnStr;
    }
	
}
