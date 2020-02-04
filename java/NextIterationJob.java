package cz.ivoa.vocloud.view;

import javax.ws.rs.*;
import java.io.IOException;
import javax.ejb.EJB;

/**
 * Created by tma on 30.1.20
 */

@Path("/job")
public class NextIterationJob {

    @EJB
    private CreateJobBean cjb;

    @Path("/iterate")
    @POST
    @Consumes("text/plain")
    @Produces("text/plain")
    public String postMessage (String configurationJson, @QueryParam("path") String path) {
        cjb.setConfigurationJson(configurationJson);
        return "ok";
    }
}
